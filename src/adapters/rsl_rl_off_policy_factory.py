"""Construct the retained SAC/PMD stacks at the Isaac/RSL boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch

from adapters.off_policy import PmdAdapter, SacAdapter
from adapters.rsl_rl_pre_reset_tap import IsaacPreResetTap
from adapters.rsl_rl_transition_collector import TransitionCollector
from algorithms.off_policy_common import BehaviorCloningLoss
from algorithms.pmd import Pmd
from algorithms.sac import Sac
from algorithms.state_dependent_tanh_actor import (
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
    policy_observation_tensor,
)
from algorithms.value_backends import (
    ActionValueBackend,
    DirectActionValue,
    SuccessorActionValue,
)
from config.off_policy import (
    DirectQCfg,
    OffPolicyAlgorithmCfg,
    PmdCfg,
    SacCfg,
    SuccessorQCfg,
)
from data.demonstration_transitions import load_demonstration_rollout
from reward_model.fixed import FixedLinearReward
from reward_model.scorer import (
    LearnedLinearRewardScorer,
    LinearRewardScorer,
    RewardScorer,
)
from storage.observations import critic_observation, detached_observation
from storage.transition_replay import TransitionReplayBuffer


@dataclass(slots=True)
class OffPolicyStack:
    """Objects owned by fixed-reward or learned-reward training runners."""

    algorithm: SacAdapter | PmdAdapter
    reward: RewardScorer
    fixed_reward: FixedLinearReward | None
    feature_tap: IsaacPreResetTap


def build_off_policy_stack(
    *,
    env: Any,
    observations: Any,
    algorithm_cfg: OffPolicyAlgorithmCfg,
    ignored_reward_terms: set[str],
    force_include_reward_terms: set[str] = frozenset(),
    device: str,
    seed: int,
    reward: RewardScorer | None = None,
    bc: BehaviorCloningLoss | None = None,
) -> OffPolicyStack:
    """Build SAC/PMD and only the configured direct/successor value model."""
    torch.manual_seed(int(seed))
    compute_device = torch.device(device)
    feature_tap = IsaacPreResetTap(
        env,
        ignored_terms=ignored_reward_terms,
        force_include_terms=force_include_reward_terms,
        device=device,
        capture_terminal_observations=True,
    )
    feature_tap.install()
    fixed_reward: FixedLinearReward | None = None
    if reward is None:
        fixed_reward = FixedLinearReward(
            feature_tap.spec.names or (),
            feature_tap.weights,
            step_dt=feature_tap.step_dt,
        ).to(compute_device)
        reward = fixed_reward
    if reward.feature_dim != feature_tap.spec.dim:
        raise ValueError(
            "Reward scorer and tapped feature dimensions differ: "
            f"{reward.feature_dim} != {feature_tap.spec.dim}."
        )

    initial_observations = detached_observation(observations, compute_device)
    policy_values = policy_observation_tensor(initial_observations)
    critic_values = critic_observation(initial_observations).flatten(start_dim=1)
    if policy_values.ndim != 2 or critic_values.ndim != 2:
        raise ValueError("Expected initial actor and critic observations [N,O].")
    policy_cfg = algorithm_cfg.policy
    actor_model = StateDependentTanhGaussianActor(
        int(policy_values.shape[-1]),
        int(env.action_dim),
        hidden_dims=policy_cfg.hidden_dims,
        action_bound=policy_cfg.action_bound,
        initial_std=policy_cfg.initial_std,
        minimum_log_std=math.log(policy_cfg.minimum_std),
        maximum_log_std=math.log(policy_cfg.maximum_std),
    ).to(compute_device)
    policy = StateDependentTanhPolicy(actor_model)

    value: ActionValueBackend
    if isinstance(algorithm_cfg.value, DirectQCfg):
        value = DirectActionValue(
            observation_dim=int(critic_values.shape[-1]),
            action_dim=int(env.action_dim),
            reward=reward,
            cfg=algorithm_cfg.value,
            train_cfg=algorithm_cfg.train,
            device=compute_device,
            seed=seed,
        )
    elif isinstance(algorithm_cfg.value, SuccessorQCfg):
        if not isinstance(reward, (FixedLinearReward, LearnedLinearRewardScorer)):
            raise ValueError(
                "The successor backend requires an explicitly linear reward."
            )
        linear_reward = cast(LinearRewardScorer, reward)
        entropy_enabled = (
            algorithm_cfg.initial_entropy_temperature > 0.0
            if isinstance(algorithm_cfg, SacCfg)
            else algorithm_cfg.entropy_temperature > 0.0
        )
        value = SuccessorActionValue(
            observation_dim=int(critic_values.shape[-1]),
            action_dim=int(env.action_dim),
            feature_dim=reward.feature_dim,
            reward=linear_reward,
            cfg=algorithm_cfg.value,
            train_cfg=algorithm_cfg.train,
            use_entropy_critic=entropy_enabled,
            device=compute_device,
            seed=seed,
        )
    else:
        raise TypeError(f"Unsupported value config {type(algorithm_cfg.value)!r}.")

    replay_cfg = algorithm_cfg.train.replay
    replay = TransitionReplayBuffer(
        capacity_transitions=replay_cfg.capacity,
        device=replay_cfg.resolve_storage_device(compute_device),
        demonstration_fraction=(
            replay_cfg.demonstration_fraction
            if replay_cfg.demonstration_path is not None
            else 0.0
        ),
    )
    if replay_cfg.demonstration_path is not None:
        if feature_tap.spec.names is None:
            raise ValueError(
                "Seeding demonstrations needs named reward features to line the "
                "recorded ones up with the environment's."
            )
        observations = env.get_observations()
        replay.seed_with_demonstrations(
            load_demonstration_rollout(
                replay_cfg.demonstration_path,
                feature_names=feature_tap.spec.names,
                observation_template=(
                    observations[0] if isinstance(observations, tuple) else observations
                ),
                device=compute_device,
                max_episodes=replay_cfg.demonstration_episodes,
            )
        )
    collector = TransitionCollector(
        env,
        policy,
        feature_tap,
        rollout_steps=algorithm_cfg.train.rollout_steps,
        device=compute_device,
        seed=seed,
        action_bound=policy_cfg.action_bound,
    )
    if isinstance(algorithm_cfg, SacCfg):
        learner = Sac(
            policy,
            value,
            cfg=algorithm_cfg,
            action_dim=int(env.action_dim),
            device=compute_device,
            bc=bc,
            seed=seed,
        )
        algorithm: SacAdapter | PmdAdapter = SacAdapter(
            collector=collector,
            replay=replay,
            learner=learner,
            train_cfg=algorithm_cfg.train,
            fixed_reward=fixed_reward,
        )
    elif isinstance(algorithm_cfg, PmdCfg):
        learner = Pmd(
            policy,
            value,
            cfg=algorithm_cfg,
            device=compute_device,
            bc=bc,
            seed=seed,
        )
        algorithm = PmdAdapter(
            collector=collector,
            replay=replay,
            learner=learner,
            train_cfg=algorithm_cfg.train,
            fixed_reward=fixed_reward,
            critic_updates_per_projection=algorithm_cfg.critic_updates_per_projection,
        )
    else:
        raise TypeError(f"Unsupported algorithm config {type(algorithm_cfg)!r}.")
    return OffPolicyStack(
        algorithm=algorithm,
        reward=reward,
        fixed_reward=fixed_reward,
        feature_tap=feature_tap,
    )
