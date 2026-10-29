"""Shared builders and environment doubles for the off-policy test modules.

The environment double lives here rather than in one test file because it is a
statement of the contract `src/adapters/` relies on - both ending flags, the
pre-reset observation pass, and the finite-horizon flag. One copy means a
contract change is answered once, instead of being weakened wherever it bites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


NUM_ENVS, OBS_DIM, ACTION_DIM = 4, 5, 2


@dataclass(slots=True)
class FakeRewardTermCfg:
    weight: float
    params: dict | None
    func: Any


class FakeRewardManager:
    """Reward manager stub with the two private lists the tap reads."""

    def __init__(self) -> None:
        self._term_names = ["alive", "effort"]
        self._term_cfgs = [
            FakeRewardTermCfg(1.0, None, lambda env: torch.ones(NUM_ENVS)),
            FakeRewardTermCfg(-0.5, None, lambda env: torch.full((NUM_ENVS,), 0.2)),
        ]
        self.installed_dt: float | None = None

    def compute(self, dt: float) -> torch.Tensor:
        """Isaac's semantics: `_step_reward` holds func * weight, the sum is scaled by dt."""
        self.installed_dt = dt
        self._step_reward = torch.stack(
            [cfg.func(None) * cfg.weight for cfg in self._term_cfgs], dim=1
        )
        return self._step_reward.sum(dim=1) * dt


class FakeTerminationManager:
    """Both ending flags, because a timeout and a failure are not the same event."""

    def __init__(self) -> None:
        self.time_outs = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.terminated = torch.zeros(NUM_ENVS, dtype=torch.bool)


class FakeObservationManager:
    """The extra pre-reset pass the tap makes to capture true successors."""

    def compute(self):
        return {
            "policy": torch.zeros(NUM_ENVS, OBS_DIM),
            "critic": torch.zeros(NUM_ENVS, OBS_DIM),
        }


@dataclass(slots=True)
class FakeEnvCfg:
    is_finite_horizon: bool = False


class FakeUnwrappedEnv:
    def __init__(self, *, is_finite_horizon: bool = False) -> None:
        self.num_envs = NUM_ENVS
        self.cfg = FakeEnvCfg(is_finite_horizon=is_finite_horizon)
        self.reward_manager = FakeRewardManager()
        self.termination_manager = FakeTerminationManager()
        self.observation_manager = FakeObservationManager()
        self.step_dt = 0.02
        self.reset_time_outs = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.reset_terminated = torch.zeros(NUM_ENVS, dtype=torch.bool)


class FakeIsaacEnv:
    """Vectorized env stub: constant observations, reward from the manager.

    Steps in Isaac's order - flags first, then rewards, then the reset - so the
    pre-reset tap sees the same thing here that it sees in the simulator.
    """

    def __init__(self, *, is_finite_horizon: bool = False) -> None:
        self.unwrapped = FakeUnwrappedEnv(is_finite_horizon=is_finite_horizon)
        self.num_envs = NUM_ENVS
        self.action_dim = ACTION_DIM
        self.device = torch.device("cpu")
        self.max_episode_length = 4
        self.age = 0
        self.clip_actions = None
        self.step_calls = 0

    def get_observations(self):
        return {
            "policy": torch.zeros(NUM_ENVS, OBS_DIM),
            "critic": torch.zeros(NUM_ENVS, OBS_DIM),
        }

    def reset(self):
        self.age = 0
        return self.get_observations(), {}

    def step(self, actions):
        self.age += 1
        self.step_calls += 1
        timed_out = torch.full(
            (NUM_ENVS,), self.age >= self.max_episode_length, dtype=torch.bool
        )
        endings = self.unwrapped.termination_manager
        endings.time_outs = timed_out
        endings.terminated = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.unwrapped.reset_time_outs = timed_out
        self.unwrapped.reset_terminated = endings.terminated
        reward = self.unwrapped.reward_manager.compute(self.unwrapped.step_dt)
        if bool(timed_out.any()):
            self.age = 0
        return (
            self.get_observations(),
            reward,
            timed_out | endings.terminated,
            {"time_outs": timed_out},
        )
