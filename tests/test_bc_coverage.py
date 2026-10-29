"""BC coverage means repeated complete trials on the same initial states."""

from types import SimpleNamespace

import pytest
import torch
from conftest import build_rollout

from storage.episode_batch import EpisodeBatch
from training.bc_coverage import (
    aggregate_coverage,
    coverage_curve,
    evaluate_bc_coverage,
    sample_member_schedule,
    sampled_member_actions,
)


def test_coverage_curve_counts_successful_trials_and_solved_initial_states():
    successes = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, False],
            [True, False, True],
        ]
    )
    curve = coverage_curve(successes)
    assert [point["n"] for point in curve] == [1, 2, 4]
    assert [point["successful_trials"] for point in curve] == [1, 2, 4]
    assert [point["covered_initial_states"] for point in curve] == [1, 2, 3]
    assert curve[-1]["trial_success_fraction"] == pytest.approx(4 / 12)
    assert curve[-1]["at_least_one_success_fraction"] == 1.0


def test_without_replacement_tries_every_member_before_repeating():
    schedule = sample_member_schedule(
        member_count=5,
        num_envs=16,
        trials=8,
        seed=8801,
        mode="without_replacement",
    )
    assert schedule.shape == (8, 16)
    for env_index in range(16):
        assert sorted(schedule[:5, env_index].tolist()) == list(range(5))
        assert len(set(schedule[5:, env_index].tolist())) == 3
    torch.testing.assert_close(
        schedule,
        sample_member_schedule(
            member_count=5,
            num_envs=16,
            trials=8,
            seed=8801,
            mode="without_replacement",
        ),
    )
    with pytest.raises(ValueError, match="Unknown BC member sampling mode"):
        sample_member_schedule(
            member_count=5, num_envs=16, trials=8, seed=8801, mode="unknown"
        )


def test_with_replacement_preserves_the_previous_assignment_stream():
    schedule = sample_member_schedule(
        member_count=5, num_envs=3, trials=4, seed=8801, mode="with_replacement"
    )
    for trial_index in range(4):
        expected = torch.randint(
            5,
            (3,),
            generator=torch.Generator().manual_seed(
                8801 + 300_007 + 10_003 * trial_index
            ),
        )
        torch.testing.assert_close(schedule[trial_index], expected)


def test_coverage_curve_can_report_one_complete_member_cycle():
    curve = coverage_curve(
        torch.zeros(8, 2, dtype=torch.bool), additional_prefixes=(5,)
    )
    assert [point["n"] for point in curve] == [1, 2, 4, 5, 8]


def test_sampled_members_stay_distinct_per_environment():
    def member(value):
        def sample(observations, *, generator):
            count = observations.shape[0]
            return torch.full((count, 2), value), torch.zeros(count)

        return SimpleNamespace(
            model=SimpleNamespace(action_dim=2),
            sample_with_log_probability=sample,
        )

    members = [member(1.0), member(2.0)]
    assignments = torch.tensor([1, 0, 1])
    actions = sampled_member_actions(
        torch.zeros(3, 4),
        members=members,
        member_by_env=assignments,
        generators=[torch.Generator(), torch.Generator()],
    )
    torch.testing.assert_close(
        actions, torch.tensor([[2.0, 2.0], [1.0, 1.0], [2.0, 2.0]])
    )
    with pytest.raises(ValueError, match="outside the member index"):
        sampled_member_actions(
            torch.zeros(3, 4),
            members=members,
            member_by_env=torch.tensor([0, 2, 1]),
            generators=[torch.Generator(), torch.Generator()],
        )


@pytest.mark.parametrize("member_sampling", ["with_replacement", "without_replacement"])
def test_coverage_evaluator_uses_matched_starts_and_never_inserts_replay(
    member_sampling,
):
    calls = []
    outcomes = (
        [True, False],  # baseline, first trial
        [False, True],  # ensemble, first trial
        [False, True],  # baseline, second trial
        [True, False],  # ensemble, second trial
    )

    def collect(**kwargs):
        call_index = len(calls)
        calls.append("collect")
        actions = kwargs["action_fn"](torch.zeros(2, 3))
        assert actions.shape == (2, 2)
        rollout = build_rollout(num_envs=2, horizon=1, feature_dim=1, seed=3)
        rollout.features[:, 0, 0] = torch.tensor(
            outcomes[call_index], dtype=torch.float
        )
        rollout.truncated[:, 0] = True
        rollout.bootstrap_valid[:, 0] = False
        return EpisodeBatch(rollout, torch.ones(2, 1, dtype=torch.bool), 2)

    def sample(observations, *, generator):
        return torch.zeros(observations.shape[0], 2), torch.zeros(observations.shape[0])

    policy = SimpleNamespace(
        model=SimpleNamespace(action_dim=2), sample_with_log_probability=sample
    )
    algorithm = SimpleNamespace(
        replay=SimpleNamespace(transition_count=0, demonstration_count=0),
        collector=SimpleNamespace(
            collect=collect,
            steps=SimpleNamespace(
                num_envs=2,
                device=torch.device("cpu"),
                feature_tap=SimpleNamespace(
                    spec=SimpleNamespace(names=("terminal_success",))
                ),
            ),
        ),
    )
    seeded = []
    result = evaluate_bc_coverage(
        algorithm,
        env=SimpleNamespace(seed=lambda seed: seeded.append(seed)),
        baseline=policy,
        members=[policy, policy],
        episodes_per_seed=2,
        seeds=[123],
        trials=2,
        member_sampling=member_sampling,
    )
    assert seeded == [123] * 4
    assert calls == ["collect"] * 4
    assert result["evaluation_interactions"] == 8
    summary = aggregate_coverage(result["rows"])
    assert summary["baseline"][-1]["covered_initial_states"] == 2
    assert summary["ensemble"][-1]["covered_initial_states"] == 2
    assert all(
        trial["max_initial_observation_difference"] == 0
        for condition in ("baseline", "ensemble")
        for trial in result["rows"][0][condition]
    )
    assert algorithm.replay.transition_count == 0
