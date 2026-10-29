# ruff: noqa: E402
"""Does PMD actually improve the policy against its critic?

The evaluation test next door shows the critic recovers Q^pi. This one closes
the loop: given a critic that can be fit, the exponential target and the
bounded projection have to move the actor toward better actions.

The problem is a one-step bandit with a known optimum, so the whole question is
whether the actor follows the value it is handed. Episodes terminate after one
step, which takes bootstrapping out of it - a policy that fails to improve here
is failing at policy improvement, not at value propagation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from algorithms.off_policy.actor import (
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
)
from algorithms.off_policy.pmd import Pmd
from algorithms.off_policy.values.backends import DirectActionValue
from config.schema import (
    DirectQCfg,
    OffPolicyTrainCfg,
    PmdCfg,
    TanhGaussianPolicyCfg,
    TransitionReplayCfg,
)
from reward.fixed import FixedLinearReward
from storage.transition_replay import TransitionReplayBuffer, TransitionRollout

OBSERVATION_DIM = 2
BEST_ACTION = 0.5
NUM_TRANSITIONS = 512


def _reward_features(actions: torch.Tensor) -> torch.Tensor:
    """One feature: the negative squared error from the best action. [...,1]."""
    return -((actions - BEST_ACTION) ** 2)


def _bandit_rollout(seed: int) -> TransitionRollout:
    """Random actions and what they earn, as one-step episodes. [N,1,...]."""
    generator = torch.Generator().manual_seed(seed)
    actions = torch.rand(NUM_TRANSITIONS, 1, 1, generator=generator) * 2.0 - 1.0
    observations = torch.zeros(NUM_TRANSITIONS, 1, OBSERVATION_DIM)
    features = _reward_features(actions)
    terminated = torch.ones(NUM_TRANSITIONS, 1, dtype=torch.bool)
    return TransitionRollout(
        policy_observations=observations,
        next_policy_observations=observations.clone(),
        critic_observations=observations.clone(),
        next_critic_observations=observations.clone(),
        actions=actions,
        features=features,
        environment_rewards=features.squeeze(-1),
        terminated=terminated,
        truncated=torch.zeros_like(terminated),
        # One-step episodes: nothing to bootstrap from, so Q(s,a) is r(a).
        bootstrap_valid=~terminated,
        policy_version=0,
        rollout_id=0,
    )


def _build_learner() -> tuple[Pmd, TransitionReplayBuffer]:
    torch.manual_seed(0)
    train_cfg = OffPolicyTrainCfg(
        gamma=0.99,
        critic_batch_size=128,
        critic_learning_rate=3.0e-3,
        target_polyak=0.05,
        replay=TransitionReplayCfg(learning_starts=1, capacity=NUM_TRANSITIONS),
    )
    policy = StateDependentTanhPolicy(
        StateDependentTanhGaussianActor(
            OBSERVATION_DIM, 1, hidden_dims=(32, 32), action_bound=1.0
        )
    )
    value = DirectActionValue(
        observation_dim=OBSERVATION_DIM,
        action_dim=1,
        reward=FixedLinearReward(("error",), torch.tensor([1.0]), step_dt=1.0),
        cfg=DirectQCfg(num_heads=2, width=64, residual_blocks=1, target_subset_size=2),
        train_cfg=train_cfg,
        step_dt=1.0,
        device=torch.device("cpu"),
        seed=0,
    )
    learner = Pmd(
        policy,
        value,
        cfg=PmdCfg(
            train=train_cfg,
            policy=TanhGaussianPolicyCfg(hidden_dims=(32, 32), action_bound=1.0),
            value=DirectQCfg(num_heads=2, width=64, residual_blocks=1),
            candidate_actions=32,
            actor_batch_size=128,
            actor_learning_rate=3.0e-3,
            projection_steps=4,
            # A wide trust region: this is about direction, not step size.
            candidate_target_kl=0.3,
            actor_projection_kl=0.1,
            actor_spread_projection_kl=0.05,
        ),
        step_dt=1.0,
        device=torch.device("cpu"),
        seed=0,
    )
    replay = TransitionReplayBuffer(
        capacity_transitions=NUM_TRANSITIONS, device=torch.device("cpu")
    )
    replay.add(_bandit_rollout(seed=0))
    return learner, replay


def _deterministic_action(learner: Pmd) -> float:
    with torch.no_grad():
        action = learner.policy.act_inference(torch.zeros(1, OBSERVATION_DIM))
    return float(action.reshape(-1)[0])


def test_pmd_moves_the_policy_toward_the_action_its_critic_prefers() -> None:
    learner, replay = _build_learner()
    before = _deterministic_action(learner)

    # Fit the critic first, so the projection is aimed at a value worth following.
    for _ in range(600):
        learner.critic_update(replay)
    for _ in range(40):
        for _ in range(4):
            learner.critic_update(replay)
        learner.actor_update(replay)

    after = _deterministic_action(learner)
    assert abs(after - BEST_ACTION) < abs(before - BEST_ACTION), (
        f"policy moved away from the optimum: {before:.3f} -> {after:.3f} "
        f"(best {BEST_ACTION})"
    )
    assert abs(after - BEST_ACTION) < 0.2, f"policy settled at {after:.3f}"


def test_every_projection_stays_inside_its_trust_region() -> None:
    """The bound is the whole point: an unbounded step is not mirror descent."""
    learner, replay = _build_learner()
    for _ in range(200):
        learner.critic_update(replay)

    for _ in range(10):
        metrics = learner.actor_update(replay)
        assert (
            metrics["PMD/actor_projection_kl"] <= learner.cfg.actor_projection_kl + 1e-6
        )
        assert metrics["PMD/actor_spread_kl"] <= (
            learner.cfg.actor_spread_projection_kl + 1e-6
        )


def test_the_frozen_policy_is_replaced_once_per_projection() -> None:
    """PMD evaluates pi_k, then improves on it; the two must stay in step."""
    learner, replay = _build_learner()
    for _ in range(50):
        learner.critic_update(replay)

    versions = []
    for _ in range(3):
        learner.actor_update(replay)
        versions.append(learner.policy_version)

    assert versions == [1, 2, 3]
