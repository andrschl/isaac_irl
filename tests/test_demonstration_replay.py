"""Demonstration-seeded replay: loading, mixing, and never evicting."""

from __future__ import annotations

import pytest
import torch
from conftest import build_rollout

from data.feature_semantics import demonstration_metadata
from storage.transition_replay import TransitionReplayBuffer

h5py = pytest.importorskip("h5py")

FEATURES = ("reaching_object", "lifting_object")
STORED_FEATURES = (*FEATURES, "terminal_success")
HORIZON = 5
STEP_DT = 0.02


def _write_demos(
    path,
    *,
    episodes: int = 3,
    horizon: int = HORIZON,
    end_at: int | None = None,
    extra_end_at: int | None = None,
    terminates: bool = False,
    is_finite_horizon: bool = False,
    metadata: dict | None = None,
    omit: tuple[str, ...] = (),
):
    """Write demonstrations in the format the recorder produces."""
    attributes = (
        demonstration_metadata(
            horizon=horizon,
            step_dt=STEP_DT,
            is_finite_horizon=is_finite_horizon,
            action_bound=4.0,
            feature_names=STORED_FEATURES,
        )
        if metadata is None
        else metadata
    )
    ends_at = horizon - 1 if end_at is None else end_at
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for key, value in attributes.items():
            data.attrs[key] = value
        for episode in range(episodes):
            demo = data.create_group(f"demo_{episode}")
            base = float(episode)
            steps = base + torch.arange(horizon * 3).reshape(horizon, 3)
            if "obs" not in omit:
                demo.create_group("obs").create_dataset("policy", data=steps.numpy())
            if "next_obs" not in omit:
                demo.create_group("next_obs").create_dataset(
                    "policy", data=(steps + 100.0).numpy()
                )
            demo.create_dataset("actions", data=torch.full((horizon, 2), base).numpy())
            demo.create_dataset("rewards", data=torch.full((horizon,), base).numpy())
            terminated = torch.zeros(horizon, dtype=torch.bool)
            truncated = torch.zeros(horizon, dtype=torch.bool)
            (terminated if terminates else truncated)[ends_at] = True
            if extra_end_at is not None:
                truncated[extra_end_at] = True
            if "terminated" not in omit:
                demo.create_dataset("terminated", data=terminated.numpy())
            if "truncated" not in omit:
                demo.create_dataset("truncated", data=truncated.numpy())
            group = demo.create_group("features")
            for index, name in enumerate(STORED_FEATURES):
                group.create_dataset(
                    name, data=torch.full((horizon,), float(index)).numpy()
                )
    return str(path)


def _load(path, **kwargs):
    from data.demonstration_transitions import load_demonstration_rollout

    settings = dict(
        feature_names=FEATURES,
        observation_template=torch.zeros(1, 3),
        device="cpu",
        horizon=HORIZON,
        step_dt=STEP_DT,
        is_finite_horizon=False,
    )
    settings.update(kwargs)
    return load_demonstration_rollout(path, **settings)


def test_every_recorded_transition_survives_the_round_trip(tmp_path) -> None:
    """The last transition of an episode is the one the sparse reward arrives on.

    An earlier format dropped it, because its successor could only be read off
    the next row. It is now recorded explicitly, so dropping it would throw away
    exactly the step the task pays for.
    """
    rollout = _load(_write_demos(tmp_path / "demos.hdf5", episodes=3))

    assert rollout.actions.shape == (3, HORIZON, 2)
    assert rollout.features.shape == (3, HORIZON, len(FEATURES))
    # Successors are read back as stored, not shifted out of the observations.
    torch.testing.assert_close(
        rollout.next_critic_observations, rollout.critic_observations + 100.0
    )
    assert bool(rollout.truncated[:, -1].all())
    assert not bool(rollout.terminated.any())


def test_the_bootstrap_mask_follows_this_run_s_episode_definition(tmp_path) -> None:
    """The same recorded time limit means two different things, so it is asked."""
    continuing = _load(_write_demos(tmp_path / "continuing.hdf5"))
    finite = _load(
        _write_demos(tmp_path / "finite.hdf5", is_finite_horizon=True),
        is_finite_horizon=True,
    )

    assert bool(continuing.bootstrap_valid.all())
    assert not bool(finite.bootstrap_valid[:, -1].any())
    assert bool(finite.bootstrap_valid[:, :-1].all())


def test_a_recorded_failure_never_bootstraps(tmp_path) -> None:
    rollout = _load(_write_demos(tmp_path / "demos.hdf5", terminates=True))

    assert bool(rollout.terminated[:, -1].all())
    assert not bool(rollout.bootstrap_valid[:, -1].any())


def test_loader_orders_features_by_the_names_it_is_given(tmp_path) -> None:
    path = _write_demos(tmp_path / "demos.hdf5")

    forward = _load(path, feature_names=FEATURES)
    reversed_order = _load(path, feature_names=tuple(reversed(FEATURES)))

    torch.testing.assert_close(forward.features, reversed_order.features.flip(-1))


def test_one_recording_serves_an_arm_that_learns_over_fewer_columns(tmp_path) -> None:
    """Extra stored columns are fine; a requested one that is absent is not.

    The arm that holds the sparse coefficient fixed fits a strictly smaller
    basis than the recording stores. Re-recording for it would make it a
    different dataset, which is the one thing the comparison cannot afford.
    """
    path = _write_demos(tmp_path / "demos.hdf5")

    subset = _load(path, feature_names=("lifting_object",))

    assert subset.features.shape[-1] == 1
    with pytest.raises(ValueError, match="missing reward features"):
        _load(path, feature_names=(*FEATURES, "object_goal_tracking"))


def test_loader_rejects_a_reset_inside_an_episode(tmp_path) -> None:
    path = _write_demos(tmp_path / "demos.hdf5", extra_end_at=2)

    with pytest.raises(ValueError, match="ends an episode before its final step"):
        _load(path)


def test_loader_rejects_an_episode_that_was_cut_short(tmp_path) -> None:
    """A fragment's last stored successor is the start of somewhere else."""
    path = _write_demos(tmp_path / "demos.hdf5", end_at=2)

    with pytest.raises(ValueError, match="does not end on its last step"):
        _load(path)


@pytest.mark.parametrize("omitted", ["next_obs", "terminated", "truncated"])
def test_a_demonstration_without_terminal_information_is_refused(
    tmp_path, omitted
) -> None:
    """Nothing here can be reconstructed, so nothing here is reconstructed."""
    path = _write_demos(tmp_path / "demos.hdf5", omit=(omitted,))

    with pytest.raises(ValueError, match=f"missing \\['{omitted}'\\]"):
        _load(path)


def test_a_legacy_recording_is_rejected_rather_than_reinterpreted(tmp_path) -> None:
    path = _write_demos(tmp_path / "demos.hdf5", metadata={"total": 3})

    with pytest.raises(ValueError, match="transition semantics"):
        _load(path)


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"horizon": HORIZON + 1}, "episode_horizon"),
        ({"step_dt": STEP_DT * 2}, "step_dt"),
        ({"is_finite_horizon": True}, "is_finite_horizon"),
    ],
)
def test_demonstrations_from_a_different_episode_are_refused(
    tmp_path, override, expected
) -> None:
    path = _write_demos(tmp_path / "demos.hdf5")

    with pytest.raises(ValueError, match=expected):
        _load(path, **override)


def test_loader_matches_the_environment_observation_tree(tmp_path) -> None:
    """Isaac reports a single named group, not a bare tensor."""
    path = _write_demos(tmp_path / "demos.hdf5")

    rollout = _load(path, observation_template={"policy": torch.zeros(1, 3)})

    assert set(rollout.policy_observations) == {"policy"}
    torch.testing.assert_close(
        rollout.policy_observations["policy"], rollout.critic_observations
    )

    with pytest.raises(ValueError, match="store observation groups"):
        _load(
            path,
            observation_template={
                "policy": torch.zeros(1, 3),
                "critic": torch.zeros(1, 5),
            },
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


@pytest.mark.parametrize("heads", [1, 3, 5])
def test_contiguous_head_slices_share_the_same_replay_mixture(heads) -> None:
    replay = _seeded_replay(0.5)
    batch = replay.sample(
        heads * 256,
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu"),
    )
    demo = (batch.actions == 9.0).all(dim=-1).reshape(heads, 256)
    assert int(demo.sum()) == heads * 128
    # Shuffle gives the same distribution, not an exact quota per head.
    assert bool(((demo.float().mean(1) > 0.35) & (demo.float().mean(1) < 0.65)).all())


def test_mixture_shuffle_preserves_every_transition_field_and_rng_reproducibility():
    from dataclasses import fields

    from storage.transition_replay import TransitionBatch

    replay = TransitionReplayBuffer(capacity_transitions=16, demonstration_fraction=0.5)
    sources = []
    for source in range(2):
        rollout = build_rollout(num_envs=1, horizon=8, seed=source)
        rollout.actions[:, :, 0] = torch.arange(8) + 8 * source
        rollout.truncated[:, -1] = True
        rollout.bootstrap_valid[:, -1] = False
        # Exercise nested observations as well as every scalar/mask field.
        rollout.policy_observations = {"policy": rollout.policy_observations}
        rollout.next_policy_observations = {"policy": rollout.next_policy_observations}
        sources.append(rollout)
    replay.seed_with_demonstrations(sources[0])
    replay.add(sources[1])
    draws = [
        replay.sample(
            64, generator=torch.Generator().manual_seed(19), device=torch.device("cpu")
        )
        for _ in range(2)
    ]
    for row, identifier in enumerate(draws[0].actions[:, 0].long().tolist()):
        source, time = divmod(identifier, 8)
        for field in fields(TransitionBatch):
            expected = getattr(sources[source], field.name)
            first = getattr(draws[0], field.name)
            second = getattr(draws[1], field.name)
            if isinstance(expected, dict):
                expected, first, second = (
                    expected["policy"],
                    first["policy"],
                    second["policy"],
                )
            torch.testing.assert_close(first[row], expected[0, time])
            torch.testing.assert_close(first, second)


def test_demo_only_sampling_before_collection_does_not_count_online_data() -> None:
    replay = TransitionReplayBuffer(capacity_transitions=32, demonstration_fraction=0.5)
    demos = build_rollout(num_envs=2, horizon=4)
    demos.actions.fill_(9.0)
    replay.seed_with_demonstrations(demos)
    batch = replay.sample(
        16, generator=torch.Generator().manual_seed(0), device=torch.device("cpu")
    )
    assert bool((batch.actions == 9.0).all())
    assert replay.transition_count == 0
    assert replay.demonstration_count == 8
    replay.add(build_rollout(num_envs=4, horizon=8, seed=2))
    mixed = replay.sample(
        16, generator=torch.Generator().manual_seed(0), device=torch.device("cpu")
    )
    assert int((mixed.actions == 9.0).all(-1).sum()) == 8


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
