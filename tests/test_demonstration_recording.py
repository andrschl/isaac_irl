"""Recording demonstrations: what is kept, what is thrown away, what is written."""

# ruff: noqa: E402

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")

from data import RobomimicDataCollector
from data.demonstration_transitions import load_demonstration_rollout
from data.feature_semantics import demonstration_metadata
from storage.episode_batch import EpisodeBatch
from storage.transition_replay import TransitionRollout
from training.record import (
    RecordingCost,
    _teacher_view,
    _write_batch,
    record_demonstrations,
    success_feature_index,
)

FEATURES = ("reaching_object", "terminal_success")
HORIZON = 4
STEP_DT = 0.02
OBS_DIM, ACTION_DIM = 3, 2


def _batch(*, successes: list[bool], lengths: list[int] | None = None) -> EpisodeBatch:
    """One wave of attempts, where ``successes`` fire on the final transition."""
    count = len(successes)
    lengths = lengths or [HORIZON] * count
    steps = torch.arange(float(count * HORIZON * OBS_DIM)).reshape(
        count, HORIZON, OBS_DIM
    )
    mask = torch.arange(HORIZON)[None, :] < torch.tensor(lengths)[:, None]
    terminated = torch.zeros(count, HORIZON, dtype=torch.bool)
    truncated = torch.zeros(count, HORIZON, dtype=torch.bool)
    truncated[torch.arange(count), torch.tensor(lengths) - 1] = True
    features = torch.zeros(count, HORIZON, len(FEATURES))
    features[:, :, 0] = 1.0
    for episode, (succeeded, length) in enumerate(zip(successes, lengths)):
        if succeeded:
            features[episode, length - 1, 1] = 1.0 / STEP_DT
    rollout = TransitionRollout(
        policy_observations={"policy": steps},
        next_policy_observations={"policy": steps + 100.0},
        critic_observations=steps.clone(),
        next_critic_observations=steps + 100.0,
        actions=torch.full((count, HORIZON, ACTION_DIM), 0.5),
        features=features,
        environment_rewards=features[:, :, 1] * STEP_DT,
        terminated=terminated,
        truncated=truncated,
        bootstrap_valid=~(terminated | truncated),
        policy_version=0,
        rollout_id=0,
    )
    masked = TransitionRollout(
        **{
            field.name: getattr(rollout, field.name)
            if field.name in {"policy_version", "rollout_id"}
            else _mask(getattr(rollout, field.name), mask)
            for field in fields(TransitionRollout)
        }
    )
    return EpisodeBatch(masked, mask, count * HORIZON)


def _mask(value, mask: torch.Tensor):
    if isinstance(value, dict):
        return {key: _mask(item, mask) for key, item in value.items()}
    return value.masked_fill(~mask.reshape(*mask.shape, *([1] * (value.ndim - 2))), 0)


def _writer(tmp_path: Path, *, num_demos: int = 4, **overrides):
    settings = dict(
        horizon=HORIZON,
        step_dt=STEP_DT,
        is_finite_horizon=True,
        action_bound=4.0,
        feature_names=FEATURES,
    )
    settings.update(overrides)
    writer = RobomimicDataCollector(
        task_name="Isaac-Unit-Test-v0",
        output_dir=str(tmp_path),
        filename="demos.hdf5",
        num_demos=num_demos,
        metadata=demonstration_metadata(**settings),
    )
    writer.reset()
    return writer


def test_only_complete_successful_attempts_become_demonstrations(tmp_path) -> None:
    """An attempt that failed is not expert data, but it did cost simulator time."""
    writer = _writer(tmp_path)

    accepted = _write_batch(
        writer,
        _batch(successes=[True, False, True, True], lengths=[4, 4, 2, 4]),
        feature_names=FEATURES,
    )
    writer.close()

    # Episode 2 succeeded but ended early, so it is not a complete episode.
    assert accepted == 2
    assert writer.written_episodes == 2
    assert writer.discarded_episodes == 2


def test_what_was_recorded_is_what_loads_back(tmp_path) -> None:
    """Round trip through the file the training run will actually read."""
    writer = _writer(tmp_path)
    batch = _batch(successes=[True, True])
    _write_batch(writer, batch, feature_names=FEATURES)
    writer.close()

    rollout = load_demonstration_rollout(
        str(tmp_path / "demos.hdf5"),
        feature_names=FEATURES,
        observation_template={"policy": torch.zeros(1, OBS_DIM)},
        device="cpu",
        horizon=HORIZON,
        step_dt=STEP_DT,
        is_finite_horizon=True,
    )

    torch.testing.assert_close(rollout.actions, batch.transitions.actions)
    torch.testing.assert_close(rollout.features, batch.transitions.features)
    torch.testing.assert_close(
        rollout.policy_observations["policy"],
        batch.transitions.policy_observations["policy"],
    )
    torch.testing.assert_close(
        rollout.next_policy_observations["policy"],
        batch.transitions.next_policy_observations["policy"],
    )
    assert bool(rollout.truncated[:, -1].all())
    # A finite horizon carries no value past the end, in replay as in collection.
    assert not bool(rollout.bootstrap_valid[:, -1].any())
    assert bool(rollout.bootstrap_valid[:, :-1].all())


def test_recording_attempt_budget_stops_failed_teacher_without_extra_collection(
    tmp_path,
):
    class FailedWaves:
        steps = type("Steps", (), {"num_envs": 2})()
        calls = 0

        def collect(self, **kwargs):
            self.calls += 1
            return _batch(successes=[False, False])

    waves = FailedWaves()
    writer = _writer(tmp_path, num_demos=1)
    try:
        with pytest.raises(RuntimeError, match="2 attempts, 0 successes"):
            record_demonstrations(
                episodes=waves,
                writer=writer,
                action_fn=lambda obs: obs,
                feature_names=FEATURES,
                num_demos=1,
                horizon=HORIZON,
                max_attempted_episodes=3,
            )
        assert waves.calls == 1
        assert writer.written_episodes == 0
    finally:
        writer.close()


def test_recording_stops_at_the_demonstrations_it_was_asked_for(tmp_path) -> None:
    """Failed waves are paid for and reported, not silently repeated forever."""

    class _Waves:
        steps = type("_Steps", (), {"num_envs": 2})()

        def __init__(self) -> None:
            self.calls = 0

        def collect(self, *, policy_version, episodes, action_fn):
            del policy_version, episodes, action_fn
            self.calls += 1
            # The first wave fails outright; the next two succeed.
            return _batch(successes=[self.calls > 1, self.calls > 1])

    writer = _writer(tmp_path, num_demos=3)
    cost = record_demonstrations(
        episodes=_Waves(),
        writer=writer,
        action_fn=lambda observations: observations,
        feature_names=FEATURES,
        num_demos=3,
        horizon=HORIZON,
    )
    writer.close()

    assert cost.written_episodes == 3
    assert cost.attempted_episodes == 6
    assert cost.successful_episodes == 4
    # A teacher that fails and a teacher that overshoots its quota look the
    # same in a single 'discarded' count, and they mean opposite things.
    assert cost.failed_episodes == 2
    assert cost.surplus_episodes == 1
    assert cost.success_rate == pytest.approx(4 / 6)
    assert cost.environment_transitions == 3 * 2 * HORIZON
    assert cost.kept_transitions == 3 * HORIZON
    assert cost.discarded_transitions == 3 * HORIZON
    assert "4 succeeded (66.7%), 2 failed, 1 succeeded past the quota" in cost.report()


def test_a_teacher_that_never_fails_is_never_reported_as_failing() -> None:
    """Every attempt qualified; the quota is what threw the rest away."""
    cost = RecordingCost(
        environment_transitions=16_000,
        attempted_episodes=64,
        successful_episodes=64,
        written_episodes=10,
        horizon=250,
    )

    assert cost.success_rate == 1.0
    assert cost.failed_episodes == 0
    assert cost.surplus_episodes == 54
    assert "0 failed" in cost.report()


def test_an_existing_demonstration_file_is_not_quietly_replaced(tmp_path) -> None:
    _writer(tmp_path).close()

    with pytest.raises(FileExistsError, match="already holds demonstrations"):
        _writer(tmp_path)

    RobomimicDataCollector(
        task_name="Isaac-Unit-Test-v0",
        output_dir=str(tmp_path),
        filename="demos.hdf5",
        num_demos=1,
        overwrite=True,
    ).reset()


def test_a_task_with_no_success_term_cannot_be_recorded_against() -> None:
    assert success_feature_index(FEATURES) == 1
    assert success_feature_index(("a", "success_bonus")) == 1

    with pytest.raises(ValueError, match="no success to record against"):
        success_feature_index(("reaching_object",))


def test_the_teacher_never_sees_the_episode_clock_it_was_not_trained_on() -> None:
    """The student's observation grew; the teacher's weights did not."""
    observations = {"policy": torch.arange(10.0).reshape(1, 10)}

    view = _teacher_view(observations, {"policy": 8})

    assert tuple(view["policy"].shape) == (1, 8)
    torch.testing.assert_close(view["policy"], observations["policy"][:, :8])
