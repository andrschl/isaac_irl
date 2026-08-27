"""Shared builders for the off-policy (SAC/PMD) test modules."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from storage.transition_replay import (  # noqa: E402
    TransitionReplayBuffer,
    TransitionRollout,
)


def build_rollout(
    *,
    num_envs: int = 4,
    horizon: int = 8,
    observation_dim: int = 3,
    action_dim: int = 2,
    feature_dim: int = 2,
    rollout_id: int = 0,
    policy_version: int = 0,
    seed: int = 0,
) -> TransitionRollout:
    """Build one valid rollout chunk with distinguishable per-step values.

    Shapes: observations [N,T+1,O], actions [N,T,A], features [N,T,D].
    """
    generator = torch.Generator().manual_seed(seed)
    trajectory = torch.randn(
        num_envs, horizon + 1, observation_dim, generator=generator
    )
    observations = trajectory[:, :-1].contiguous()
    next_observations = trajectory[:, 1:].contiguous()
    actions = torch.randn(num_envs, horizon, action_dim, generator=generator).tanh()
    features = torch.randn(num_envs, horizon, feature_dim, generator=generator)
    return TransitionRollout(
        policy_observations=observations,
        next_policy_observations=next_observations,
        critic_observations=observations.clone(),
        next_critic_observations=next_observations.clone(),
        actions=actions,
        features=features,
        environment_rewards=features.sum(dim=-1),
        terminated=torch.zeros(num_envs, horizon, dtype=torch.bool),
        truncated=torch.zeros(num_envs, horizon, dtype=torch.bool),
        bootstrap_valid=torch.ones(num_envs, horizon, dtype=torch.bool),
        policy_version=policy_version,
        rollout_id=rollout_id,
    )


def build_replay(rollouts: list[TransitionRollout]) -> TransitionReplayBuffer:
    """Return a replay buffer holding ``rollouts`` in order."""
    replay = TransitionReplayBuffer(capacity_transitions=1_000_000)
    for rollout in rollouts:
        replay.add(rollout)
    return replay


@pytest.fixture
def rollout_builder():
    return build_rollout


@pytest.fixture
def replay_builder():
    return build_replay
