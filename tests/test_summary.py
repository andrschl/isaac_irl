"""The summary contract: one small metric set for every algorithm."""

from __future__ import annotations

import pytest

from runner.summary import TRANSITIONS, SampleEfficiencyTracker, training_summary


def test_summary_selects_the_same_keys_for_a_successor_q_run() -> None:
    summary = training_summary(
        {
            "SuccessorQ/td_loss": 0.25,
            "Value/q_mean": 3.0,
            "Actor/policy_std_mean": 0.5,
            "PMD/actor_projection_kl": 0.02,
            "Data/actor_updates": 16.0,
            "Train/reward_per_step": 0.7,
            "IRL/reward_loss": -1.5,
            "IRL/feature_exp_diff_norm": 0.4,
        }
    )

    assert summary == {
        "Summary/critic_loss": 0.25,
        "Summary/q_mean": 3.0,
        "Summary/policy_std": 0.5,
        "Summary/actor_kl": 0.02,
        "Summary/policy_updates": 16.0,
        "Summary/train_reward_per_step": 0.7,
        "Summary/reward_loss": -1.5,
        "Summary/feature_gap_norm": 0.4,
    }


def test_the_same_summary_keys_come_from_direct_q_and_from_ppo() -> None:
    """A dashboard is only comparable if every algorithm fills the same slots."""
    direct = training_summary({"DirectQ/td_loss": 0.5, "Actor/policy_std_mean": 0.3})
    ppo = training_summary({"RL/value_loss": 0.5, "RL/policy_std_mean": 0.3})

    assert direct == ppo == {"Summary/critic_loss": 0.5, "Summary/policy_std": 0.3}


def test_bc_reports_one_loss_whichever_algorithm_produced_it() -> None:
    assert training_summary({"BC/loss": 0.2}) == {"Summary/bc_loss": 0.2}
    assert training_summary({"Actor/bc_loss": 0.2}) == {"Summary/bc_loss": 0.2}


def test_summary_omits_what_an_algorithm_did_not_report() -> None:
    assert training_summary({"Data/replay_transitions": 10.0}) == {}


def test_tracker_reports_best_and_area_under_the_return_curve() -> None:
    tracker = SampleEfficiencyTracker(validation_steps=10)

    first = tracker.update(
        simulator_transitions=100, environment_return_mean=5.0, success_rate=0.1
    )
    second = tracker.update(
        simulator_transitions=200, environment_return_mean=3.0, success_rate=0.4
    )

    assert first["Summary/return_per_step"] == pytest.approx(0.5)
    assert second["Summary/return_per_step"] == pytest.approx(0.3)
    assert second["Summary/best_return_per_step"] == pytest.approx(0.5)
    assert second["Summary/best_success_rate"] == pytest.approx(0.4)
    # Trapezoid between (100, 0.5) and (200, 0.3), averaged over 100 transitions.
    assert second["Summary/return_per_step_auc"] == pytest.approx(0.4)


def test_tracker_records_the_first_transition_count_that_hit_a_threshold() -> None:
    tracker = SampleEfficiencyTracker(
        validation_steps=10, success_rate_thresholds=(0.5,)
    )

    tracker.update(
        simulator_transitions=100, environment_return_mean=1.0, success_rate=0.2
    )
    crossed = tracker.update(
        simulator_transitions=200, environment_return_mean=1.0, success_rate=0.6
    )
    later = tracker.update(
        simulator_transitions=300, environment_return_mean=1.0, success_rate=0.9
    )

    key = "Summary/transitions_to_success_rate_0_5"
    assert crossed[key] == 200.0
    assert later[key] == 200.0, "the threshold records when it was first reached"


def test_tracker_refuses_validations_that_do_not_advance() -> None:
    tracker = SampleEfficiencyTracker(validation_steps=10)
    tracker.update(
        simulator_transitions=100, environment_return_mean=1.0, success_rate=None
    )

    with pytest.raises(ValueError, match="must increase strictly"):
        tracker.update(
            simulator_transitions=100, environment_return_mean=1.0, success_rate=None
        )


def test_transitions_is_the_shared_x_axis_name() -> None:
    assert TRANSITIONS == "Summary/transitions"
