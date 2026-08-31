"""Demonstration-seeded replay: loading, mixing, and never evicting."""

from __future__ import annotations

import pytest
import torch

from conftest import build_rollout
from storage.transition_replay import TransitionReplayBuffer

h5py = pytest.importorskip("h5py")

FEATURES = ("reaching_object", "lifting_object")


def _write_demos(
    path, *, episodes: int = 3, horizon: int = 5, done_at: int | None = None
):
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for episode in range(episodes):
            demo = data.create_group(f"demo_{episode}")
            base = float(episode)
            demo.create_group("obs").create_dataset(
                "policy",
                data=base + torch.arange(horizon * 3).reshape(horizon, 3).numpy(),
            )
            demo.create_dataset("actions", data=torch.full((horizon, 2), base).numpy())
            demo.create_dataset("rewards", data=torch.full((horizon,), base).numpy())
            dones = torch.zeros(horizon, dtype=torch.bool)
            dones[-1 if done_at is None else done_at] = True
            demo.create_dataset("dones", data=dones.numpy())
            group = demo.create_group("features")
            for index, name in enumerate(FEATURES):
                group.create_dataset(
                    name, data=torch.full((horizon,), float(index)).numpy()
                )
    return str(path)


def _load(path, **kwargs):
    from data.demonstration_transitions import load_demonstration_rollout

    return load_demonstration_rollout(
        path,
        feature_names=FEATURES,
        observation_template=torch.zeros(1, 3),
        device="cpu",
        **kwargs,
    )


def test_loader_drops_the_step_whose_successor_is_a_reset(tmp_path) -> None:
    rollout = _load(_write_demos(tmp_path / "demos.hdf5", episodes=3, horizon=5))

    # Five recorded steps hold four transitions with a real successor.
    assert rollout.actions.shape == (3, 4, 2)
    assert rollout.features.shape == (3, 4, len(FEATURES))
    torch.testing.assert_close(
        rollout.next_policy_observations[:, :-1], rollout.policy_observations[:, 1:]
    )
    assert not rollout.terminated.any()
    assert rollout.bootstrap_valid.all()


def test_loader_orders_features_by_the_names_it_is_given(tmp_path) -> None:
    from data.demonstration_transitions import load_demonstration_rollout

    path = _write_demos(tmp_path / "demos.hdf5")
    forward = load_demonstration_rollout(
        path,
        feature_names=FEATURES,
        observation_template=torch.zeros(1, 3),
        device="cpu",
    )
    reversed_order = load_demonstration_rollout(
        path,
        feature_names=tuple(reversed(FEATURES)),
        observation_template=torch.zeros(1, 3),
        device="cpu",
    )

    torch.testing.assert_close(forward.features, reversed_order.features.flip(-1))


def test_loader_rejects_a_reset_inside_an_episode(tmp_path) -> None:
    path = _write_demos(tmp_path / "demos.hdf5", horizon=5, done_at=2)

    with pytest.raises(ValueError, match="ends an episode before its final step"):
        _load(path)


def test_loader_rejects_missing_reward_features(tmp_path) -> None:
    from data.demonstration_transitions import load_demonstration_rollout

    path = _write_demos(tmp_path / "demos.hdf5")

    with pytest.raises(ValueError, match="missing reward features"):
        load_demonstration_rollout(
            path,
            feature_names=(*FEATURES, "object_goal_tracking"),
            observation_template=torch.zeros(1, 3),
            device="cpu",
        )


def test_limits_the_episodes_it_reads(tmp_path) -> None:
    rollout = _load(_write_demos(tmp_path / "demos.hdf5", episodes=4), max_episodes=2)

    assert rollout.actions.shape[0] == 2


def _seeded_replay(fraction: float) -> TransitionReplayBuffer:
    """A replay whose demonstrations are marked by an impossible action value."""
    replay = TransitionReplayBuffer(
        capacity_transitions=32, demonstration_fraction=fraction
    )
    demonstrations = build_rollout(num_envs=2, horizon=4, seed=1)
    demonstrations.actions.fill_(9.0)
    replay.seed_with_demonstrations(demonstrations)
    replay.add(build_rollout(num_envs=4, horizon=8, seed=2))
    return replay


def test_every_batch_holds_the_configured_demonstration_share() -> None:
    replay = _seeded_replay(0.5)

    batch = replay.sample(
        16, generator=torch.Generator().manual_seed(0), device=torch.device("cpu")
    )

    from_demonstrations = (batch.actions == 9.0).all(dim=-1)
    assert batch.actions.shape[0] == 16
    assert int(from_demonstrations.sum()) == 8


def test_demonstrations_survive_the_ring_wrapping_around() -> None:
    replay = _seeded_replay(0.25)
    for rollout_id in range(3, 8):
        replay.add(
            build_rollout(num_envs=4, horizon=8, seed=rollout_id, rollout_id=rollout_id)
        )

    batch = replay.sample(
        8, generator=torch.Generator().manual_seed(0), device=torch.device("cpu")
    )

    assert replay.transition_count == 32  # the ring is full and has overwritten
    assert int((batch.actions == 9.0).all(dim=-1).sum()) == 2


def test_seeding_without_a_demonstration_share_is_rejected() -> None:
    replay = TransitionReplayBuffer(capacity_transitions=32)

    with pytest.raises(ValueError, match="demonstration_fraction"):
        replay.seed_with_demonstrations(build_rollout(num_envs=2, horizon=4))
