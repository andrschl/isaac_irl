"""Construct the PMD stack at the Isaac/RSL boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from einops import rearrange

from adapters.checkpoints import load_actor_only_checkpoint
from adapters.episode_collector import EpisodeCollector
from adapters.pmd import PmdAdapter
from adapters.pre_reset_tap import IsaacPreResetTap
from adapters.transition_collector import TransitionCollector
from algorithms.off_policy.actor import (
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
    policy_observation_tensor,
)
from algorithms.off_policy.common import BehaviorCloningLoss
from algorithms.off_policy.pmd import Pmd
from algorithms.off_policy.values.backends import (
    ActionValueBackend,
    DirectActionValue,
    SuccessorActionValue,
)
from config.schema import (
    DirectQCfg,
    OffPolicyAlgorithmCfg,
    PmdCfg,
    SuccessorQCfg,
)
from data.demonstration_transitions import load_demonstration_rollout
from reward.fixed import FixedLinearReward
from reward.scorer import (
    AdditiveLinearRewardScorer,
    LearnedLinearRewardScorer,
    LinearRewardScorer,
    RewardScorer,
)
from storage.observations import critic_observation, detached_observation
from storage.transition_replay import TransitionReplayBuffer


@dataclass(slots=True)
class OffPolicyStack:
    """Objects owned by the training runner for one off-policy algorithm."""

    algorithm: PmdAdapter
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
    init_policy_checkpoint: str | None = None,
) -> OffPolicyStack:
    """Build PMD with only the configured value representation.

    ``reward`` is what the policy maximizes; ``None`` means the environment's
    own reward, which is then represented by a fixed linear model over the
    tapped reward terms so both reward sources travel the same path.
    """
    if not isinstance(algorithm_cfg, PmdCfg):
        raise TypeError(f"Expected PmdCfg, got {type(algorithm_cfg).__name__}.")
    torch.manual_seed(int(seed))
    compute_device = torch.device(device)

    feature_tap = _build_feature_tap(
        env,
        ignored_terms=ignored_reward_terms,
        force_include_terms=force_include_reward_terms,
        device=device,
    )
    reward, fixed_reward = _resolve_reward(reward, feature_tap, compute_device)

    policy_dim, critic_dim = _observation_dims(observations, compute_device)
    policy = _build_policy(
        algorithm_cfg, observation_dim=policy_dim, env=env, device=compute_device
    )
    if init_policy_checkpoint is not None:
        load_actor_only_checkpoint(
            policy.model, init_policy_checkpoint, device=compute_device
        )
    value = _build_value(
        algorithm_cfg,
        observation_dim=critic_dim,
        action_dim=int(env.action_dim),
        reward=reward,
        step_dt=feature_tap.step_dt,
        device=compute_device,
        seed=seed,
        horizon=int(env.max_episode_length),
    )
    replay = _build_replay(
        algorithm_cfg,
        feature_tap=feature_tap,
        env=env,
        device=compute_device,
        value=value,
    )
    steps = TransitionCollector(
        env,
        policy,
        feature_tap,
        rollout_steps=1,
        device=compute_device,
        seed=seed,
        is_finite_horizon=bool(env.unwrapped.cfg.is_finite_horizon),
    )
    collector = EpisodeCollector(
        steps, episodes_per_batch=algorithm_cfg.train.episodes_per_batch
    )
    algorithm = PmdAdapter(
        collector=collector,
        replay=replay,
        learner=Pmd(
            policy,
            value,
            cfg=algorithm_cfg,
            step_dt=feature_tap.step_dt,
            device=compute_device,
            bc=bc,
            seed=seed,
        ),
        train_cfg=algorithm_cfg.train,
        fixed_reward=fixed_reward,
        policy_improvements_per_batch=algorithm_cfg.policy_improvements_per_batch,
    )
    return OffPolicyStack(
        algorithm=algorithm,
        reward=reward,
        fixed_reward=fixed_reward,
        feature_tap=feature_tap,
    )


def _build_feature_tap(
    env: Any,
    *,
    ignored_terms: set[str],
    force_include_terms: set[str],
    device: str,
) -> IsaacPreResetTap:
    """Install the tap that captures reward terms before Isaac's auto-reset."""
    tap = IsaacPreResetTap(
        env,
        ignored_terms=ignored_terms,
        force_include_terms=force_include_terms,
        device=device,
        capture_terminal_observations=True,
    )
    tap.install()
    return tap


def _resolve_reward(
    reward: RewardScorer | None,
    feature_tap: IsaacPreResetTap,
    device: torch.device,
) -> tuple[RewardScorer, FixedLinearReward | None]:
    """Decide what scores actions, and keep the fixed model when there is one.

    A fixed model is returned alongside so collection can check every rollout
    that the features really do reconstruct the environment's reward.
    """
    fixed_reward: FixedLinearReward | None = None
    if reward is None:
        fixed_reward = FixedLinearReward(
            feature_tap.spec.names or (),
            feature_tap.weights,
            step_dt=feature_tap.step_dt,
        ).to(device)
        reward = fixed_reward
    if reward.feature_dim != feature_tap.spec.dim:
        raise ValueError(
            "Reward scorer and tapped feature dimensions differ: "
            f"{reward.feature_dim} != {feature_tap.spec.dim}."
        )
    return reward, fixed_reward


def _observation_dims(observations: Any, device: torch.device) -> tuple[int, int]:
    """Actor and critic input widths, read off one observation. -> (O_actor, O_critic)."""
    initial = detached_observation(observations, device)
    policy_values = policy_observation_tensor(initial)
    critic_values = rearrange(critic_observation(initial), "n ... -> n (...)")
    if policy_values.ndim != 2:
        raise ValueError(
            f"Expected actor observations [N,O], got {tuple(policy_values.shape)}."
        )
    return int(policy_values.shape[-1]), int(critic_values.shape[-1])


def _build_policy(
    algorithm_cfg: PmdCfg, *, observation_dim: int, env: Any, device: torch.device
) -> StateDependentTanhPolicy:
    policy_cfg = algorithm_cfg.policy
    model = StateDependentTanhGaussianActor(
        observation_dim,
        int(env.action_dim),
        hidden_dims=policy_cfg.hidden_dims,
        action_bound=policy_cfg.action_bound,
        initial_std=policy_cfg.initial_std,
        minimum_log_std=math.log(policy_cfg.minimum_std),
        maximum_log_std=math.log(policy_cfg.maximum_std),
        std_parameterization=policy_cfg.std_parameterization,
    ).to(device)
    return StateDependentTanhPolicy(model)


def _build_value(
    algorithm_cfg: PmdCfg,
    *,
    observation_dim: int,
    action_dim: int,
    reward: RewardScorer,
    step_dt: float,
    device: torch.device,
    seed: int,
    horizon: int | None = None,
) -> ActionValueBackend:
    """Build the configured action-value representation, and only that one."""
    value_cfg = algorithm_cfg.value
    if isinstance(value_cfg, DirectQCfg):
        return DirectActionValue(
            observation_dim=observation_dim,
            action_dim=action_dim,
            reward=reward,
            cfg=value_cfg,
            train_cfg=algorithm_cfg.train,
            step_dt=step_dt,
            device=device,
            seed=seed,
        )
    if isinstance(value_cfg, SuccessorQCfg):
        if not isinstance(
            reward,
            (FixedLinearReward, LearnedLinearRewardScorer, AdditiveLinearRewardScorer),
        ):
            # Successor targets are reward-independent only because the reward
            # is applied at scoring time as a dot product with the weights.
            raise ValueError(
                "The successor backend requires an explicitly linear reward, got "
                f"{type(reward).__name__}."
            )
        return SuccessorActionValue(
            observation_dim=observation_dim,
            action_dim=action_dim,
            feature_dim=reward.feature_dim,
            reward=cast(LinearRewardScorer, reward),
            cfg=value_cfg,
            train_cfg=algorithm_cfg.train,
            use_entropy_critic=algorithm_cfg.entropy_rate > 0.0,
            step_dt=step_dt,
            device=device,
            seed=seed,
            horizon=horizon,
        )
    raise TypeError(f"Unsupported value config {type(value_cfg).__name__}.")


def _build_replay(
    algorithm_cfg: PmdCfg,
    *,
    feature_tap: IsaacPreResetTap,
    env: Any,
    device: torch.device,
    value: ActionValueBackend,
) -> TransitionReplayBuffer:
    """Build the replay ring, seeded with demonstrations when configured."""
    replay_cfg = algorithm_cfg.train.replay
    seeds_demonstrations = replay_cfg.demonstration_path is not None
    replay = TransitionReplayBuffer(
        capacity_transitions=replay_cfg.capacity,
        device=replay_cfg.resolve_storage_device(device),
        demonstration_fraction=(
            replay_cfg.demonstration_fraction if seeds_demonstrations else 0.0
        ),
    )
    if not seeds_demonstrations:
        return replay
    if feature_tap.spec.names is None:
        raise ValueError(
            "Seeding demonstrations needs named reward features to line the "
            "recorded ones up with the environment's."
        )
    demonstrations = load_demonstration_rollout(
        replay_cfg.demonstration_path,
        feature_names=feature_tap.spec.names,
        observation_template=env.get_observations(),
        device=device,
        # The episode this run is defined on. Demonstrations recorded under
        # a different horizon, step or ending rule are other data.
        horizon=int(env.max_episode_length),
        step_dt=feature_tap.step_dt,
        is_finite_horizon=bool(env.unwrapped.cfg.is_finite_horizon),
        max_episodes=replay_cfg.demonstration_episodes,
    )
    replay.seed_with_demonstrations(demonstrations)
    value.fit_feature_scale(demonstrations)
    return replay
