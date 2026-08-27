from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from adapters.rsl_rl_transition_collector import TransitionCollector  # noqa: E402

NUM_ENVS = 3
OBSERVATION_DIM = 2
ACTION_DIM = 1
RESET_MARKER = -1.0


class _FakeTap:
    """Pre-reset tap double: it sees the state before the env resets anything.

    ``surviving_offset`` models a stateful observation pipeline, where the extra
    pre-reset pass does not reproduce what the environment later returns.
    """

    def __init__(self, *, surviving_offset: float = 0.0) -> None:
        self.surviving_offset = surviving_offset
        self._features = torch.ones(NUM_ENVS, 1)
        self._reward = torch.ones(NUM_ENVS)
        self._terminal: torch.Tensor | None = None

    def observe(self, *, observations: torch.Tensor, timed_out: bool) -> None:
        self._terminal = (
            observations.clone() + self.surviving_offset if timed_out else None
        )

    def features(self, env: Any | None = None) -> torch.Tensor:
        del env
        return self._features

    def environment_reward(self) -> torch.Tensor:
        return self._reward

    def terminal_observations(self):
        return self._terminal


class _FakeEnv:
    """Vector env double with scriptable time-outs and terminations."""

    def __init__(
        self, tap: _FakeTap, *, time_out_step: int, terminated_step: int
    ) -> None:
        self.num_envs = NUM_ENVS
        self.device = torch.device("cpu")
        self.tap = tap
        self.time_out_step = time_out_step
        self.terminated_step = terminated_step
        self.step_index = 0

    def get_observations(self) -> torch.Tensor:
        return torch.full((NUM_ENVS, OBSERVATION_DIM), float(self.step_index))

    def reset(self) -> None:
        self.step_index = 0

    def step(self, actions: torch.Tensor):
        del actions
        self.step_index += 1
        timed_out = self.step_index == self.time_out_step
        terminated = self.step_index == self.terminated_step
        observations = torch.full((NUM_ENVS, OBSERVATION_DIM), float(self.step_index))
        self.tap.observe(observations=observations, timed_out=timed_out)
        time_outs = torch.zeros(NUM_ENVS, dtype=torch.bool)
        dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
        if timed_out:
            time_outs[0] = True
            dones[0] = True
        if terminated:
            dones[1] = True
        # Isaac hands back the post-reset observation for every done env.
        observations = observations.clone()
        observations[dones] = RESET_MARKER
        return observations, torch.ones(NUM_ENVS), dones, {"time_outs": time_outs}


class _ConstantPolicy:
    model = torch.nn.Identity()

    def sample_with_log_probability(self, observations, *, generator=None):
        del generator
        batch = observations.shape[0]
        return torch.zeros(batch, ACTION_DIM), torch.zeros(batch)


def _collector(*, time_out_step: int, terminated_step: int) -> TransitionCollector:
    tap = _FakeTap()
    env = _FakeEnv(tap, time_out_step=time_out_step, terminated_step=terminated_step)
    return TransitionCollector(
        env,
        _ConstantPolicy(),
        tap,
        rollout_steps=4,
        device="cpu",
        seed=0,
    )


def test_time_limit_keeps_the_true_terminal_successor() -> None:
    collector = _collector(time_out_step=2, terminated_step=99)

    rollout, metrics = collector.collect(policy_version=0)

    # Env 0 timed out on step 2 (index 1): its successor is the pre-reset state
    # (marked by the step counter), while the next transition starts from the
    # post-reset state.
    assert float(rollout.next_critic_observations[0, 1, 0]) == 2.0
    assert float(rollout.critic_observations[0, 2, 0]) == RESET_MARKER
    # Environments that did not reset keep the observation the env returned.
    assert float(rollout.next_critic_observations[2, 1, 0]) == 2.0
    assert bool(rollout.truncated[0, 1])
    assert bool(rollout.bootstrap_valid[0, 1])
    assert metrics["Data/truncated_fraction"] > 0.0


def test_termination_stops_the_bellman_backup() -> None:
    collector = _collector(time_out_step=99, terminated_step=3)

    rollout, metrics = collector.collect(policy_version=0)

    assert bool(rollout.terminated[1, 2])
    assert not bool(rollout.bootstrap_valid[1, 2])
    assert metrics["Data/terminated_fraction"] > 0.0
    assert metrics["Data/bootstrap_disabled_fraction"] > 0.0


def test_surviving_environments_must_observe_the_same_state_twice() -> None:
    tap = _FakeTap(surviving_offset=0.5)
    env = _FakeEnv(tap, time_out_step=2, terminated_step=99)
    collector = TransitionCollector(
        env, _ConstantPolicy(), tap, rollout_steps=4, device="cpu", seed=0
    )

    with pytest.raises(ValueError, match="observation pipeline is stateful"):
        collector.collect(policy_version=0)


def test_actions_outside_the_policy_bound_are_rejected() -> None:
    class _OutOfBoundPolicy(_ConstantPolicy):
        def sample_with_log_probability(self, observations, *, generator=None):
            del generator
            batch = observations.shape[0]
            return torch.full((batch, ACTION_DIM), 2.0), torch.zeros(batch)

    tap = _FakeTap()
    env = _FakeEnv(tap, time_out_step=99, terminated_step=99)
    collector = TransitionCollector(
        env, _OutOfBoundPolicy(), tap, rollout_steps=2, device="cpu", seed=0
    )

    with pytest.raises(ValueError, match="out-of-bound action"):
        collector.collect(policy_version=0)
