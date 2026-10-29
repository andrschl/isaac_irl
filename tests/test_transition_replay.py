from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from conftest import build_replay, build_rollout  # noqa: E402

from storage.transition_replay import (  # noqa: E402
    TransitionReplayBuffer,
    TransitionRollout,
)


def test_sample_returns_paired_transitions_and_their_successors() -> None:
    rollout = build_rollout(num_envs=3, horizon=5)
    replay = build_replay([rollout])

    batch = replay.sample(
        64, generator=torch.Generator().manual_seed(0), device=torch.device("cpu")
    )

    assert batch.critic_observations.shape == (64, 3)
    # Every sampled (s, a, s') triple must exist in the stored chunk.
    flat_current = rollout.critic_observations.reshape(-1, 3)
    flat_next = rollout.next_critic_observations.reshape(-1, 3)
    flat_actions = rollout.actions.reshape(-1, 2)
    for row in range(batch.critic_observations.shape[0]):
        matches = (flat_current == batch.critic_observations[row]).all(dim=1)
        assert matches.any()
        index = int(matches.nonzero()[0].item())
        torch.testing.assert_close(
            batch.next_critic_observations[row], flat_next[index]
        )
        torch.testing.assert_close(batch.actions[row], flat_actions[index])


def test_sample_draws_from_every_stored_chunk() -> None:
    replay = build_replay(
        [build_rollout(rollout_id=index, seed=index) for index in range(3)]
    )

    batch = replay.sample(
        600, generator=torch.Generator().manual_seed(1), device=torch.device("cpu")
    )

    assert batch.features.shape == (600, 2)
    assert replay.transition_count == 3 * 4 * 8
    assert replay.policy_version_range() == (0, 0)


def test_capacity_overwrites_the_oldest_transitions() -> None:
    replay = TransitionReplayBuffer(capacity_transitions=64)
    for index in range(3):
        replay.add(
            build_rollout(
                num_envs=4,
                horizon=8,
                rollout_id=index,
                policy_version=index,
                seed=index,
            )
        )

    assert replay.transition_count == 64
    # 96 transitions were inserted into 64 slots, so generation 0 is gone.
    assert replay.policy_version_range() == (1, 2)


def test_wrapped_writes_stay_consistent_per_transition() -> None:
    replay = TransitionReplayBuffer(capacity_transitions=48)
    for index in range(3):
        replay.add(build_rollout(num_envs=4, horizon=8, rollout_id=index, seed=index))
    last = build_rollout(num_envs=4, horizon=8, rollout_id=3, seed=3)
    replay.add(last)

    batch = replay.sample(
        512, generator=torch.Generator().manual_seed(2), device=torch.device("cpu")
    )

    # Whatever survived, every sampled row is a transition that was inserted:
    # its features and reward must still agree.
    torch.testing.assert_close(
        batch.environment_rewards, batch.features.sum(dim=-1), rtol=1e-6, atol=1e-6
    )
    assert replay.transition_count == 48


def _rebuilt(rollout: TransitionRollout, **overrides) -> TransitionRollout:
    fields = {
        "policy_observations": rollout.policy_observations,
        "next_policy_observations": rollout.next_policy_observations,
        "critic_observations": rollout.critic_observations,
        "next_critic_observations": rollout.next_critic_observations,
        "actions": rollout.actions,
        "features": rollout.features,
        "environment_rewards": rollout.environment_rewards,
        "terminated": rollout.terminated,
        "truncated": rollout.truncated,
        "bootstrap_valid": rollout.bootstrap_valid,
        "policy_version": 0,
        "rollout_id": 0,
    }
    fields.update(overrides)
    return TransitionRollout(**fields)


def test_rollout_rejects_bootstrapping_from_a_terminal_state() -> None:
    rollout = build_rollout(num_envs=2, horizon=3)
    terminated = torch.zeros(2, 3, dtype=torch.bool)
    terminated[0, 1] = True

    with pytest.raises(ValueError, match="Terminated transitions must not bootstrap"):
        _rebuilt(rollout, terminated=terminated)


def test_rollout_allows_bootstrapping_through_a_time_limit() -> None:
    rollout = build_rollout(num_envs=2, horizon=3)
    truncated = torch.zeros(2, 3, dtype=torch.bool)
    truncated[1, 2] = True

    truncating = _rebuilt(rollout, truncated=truncated)

    assert bool(truncating.bootstrap_valid[1, 2])


def test_a_failure_on_the_timeout_step_keeps_both_flags() -> None:
    """Isaac reports both, and collapsing them loses which one actually happened.

    The object can drop on the very step the horizon expires. Failure wins for
    the Bellman backup, but the episode still ended at its time limit, and a
    replay that refused the pair would force the collector to lie about one.
    """
    rollout = build_rollout(num_envs=2, horizon=3)
    terminated = torch.zeros(2, 3, dtype=torch.bool)
    truncated = torch.zeros(2, 3, dtype=torch.bool)
    terminated[0, 2] = True
    truncated[0, 2] = True
    bootstrap_valid = torch.ones(2, 3, dtype=torch.bool)
    bootstrap_valid[0, 2] = False

    both = _rebuilt(
        rollout,
        terminated=terminated,
        truncated=truncated,
        bootstrap_valid=bootstrap_valid,
    )

    assert bool(both.terminated[0, 2]) and bool(both.truncated[0, 2])
    assert not bool(both.bootstrap_valid[0, 2])


def test_rollout_rejects_mismatched_feature_shape() -> None:
    rollout = build_rollout(num_envs=2, horizon=3)

    with pytest.raises(ValueError, match=r"Expected features \[N,T,D\]"):
        _rebuilt(rollout, features=rollout.features[:, :-1])


def test_sampling_an_empty_replay_is_an_error() -> None:
    replay = TransitionReplayBuffer(capacity_transitions=128)

    with pytest.raises(RuntimeError, match="Cannot sample an empty replay"):
        replay.sample(
            8, generator=torch.Generator().manual_seed(0), device=torch.device("cpu")
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gpu_replay_returns_the_same_transitions() -> None:
    rollout = build_rollout(num_envs=3, horizon=5)
    replay = TransitionReplayBuffer(capacity_transitions=1_000, device="cuda:0")
    replay.add(rollout)

    batch = replay.sample(
        32, generator=torch.Generator().manual_seed(0), device=torch.device("cuda:0")
    )

    assert batch.critic_observations.device.type == "cuda"
    stored = rollout.critic_observations.reshape(-1, 3).to("cuda:0")
    for row in range(batch.critic_observations.shape[0]):
        assert (stored == batch.critic_observations[row]).all(dim=1).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_replay_larger_than_the_device_is_refused() -> None:
    replay = TransitionReplayBuffer(
        capacity_transitions=10_000_000_000, device="cuda:0"
    )

    with pytest.raises(ValueError, match="Lower replay.capacity"):
        replay.add(build_rollout(num_envs=2, horizon=2))
