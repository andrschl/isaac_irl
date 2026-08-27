"""Tests for the compact algorithm-independent dashboard contract."""

from __future__ import annotations

import pytest

from runner.overview import SampleEfficiencyTracker, training_overview_metrics


def test_training_overview_selects_backend_independent_metrics() -> None:
    overview = training_overview_metrics(
        {
            "SuccessorQ/td_loss": 0.25,
            "Actor/parameter_delta": 0.1,
            "Actor/policy_std_mean": 0.5,
            "PMD/actor_projection_kl": 0.02,
            "Action/clipped_fraction": 0.0,
            "Data/actor_updates": 16.0,
            "Data/truncated_fraction": 0.003,
            "Diagnostics/non_finite_count": 0.0,
        }
    )

    assert overview == {
        "Overview/critic_loss": 0.25,
        "Overview/actor_parameter_delta": 0.1,
        "Overview/policy_std": 0.5,
        "Overview/pmd_actor_kl": 0.02,
        "Overview/action_clipped_fraction": 0.0,
        "Overview/policy_updates": 16.0,
        "Overview/timeout_fraction": 0.003,
        "Overview/non_finite_count": 0.0,
    }


def test_sample_efficiency_normalizes_return_and_tracks_auc() -> None:
    tracker = SampleEfficiencyTracker(
        validation_steps=250,
        return_per_step_thresholds=(0.0, 0.012),
    )

    initial = tracker.update(
        simulator_transitions=0,
        environment_return_mean=-2.5,
        success_rate=None,
    )
    final = tracker.update(
        simulator_transitions=100_000,
        environment_return_mean=4.0,
        success_rate=None,
    )

    assert initial["Overview/return_per_step"] == pytest.approx(-0.01)
    assert final["Overview/return_per_step"] == pytest.approx(0.016)
    assert final["Overview/return_per_step_auc"] == pytest.approx(0.003)
    assert final["Overview/transitions_to_return_per_step_0"] == 100_000
    assert final["Overview/transitions_to_return_per_step_0_012"] == 100_000


def test_sample_efficiency_rejects_duplicate_validation_step() -> None:
    tracker = SampleEfficiencyTracker(validation_steps=100)
    tracker.update(
        simulator_transitions=10,
        environment_return_mean=1.0,
        success_rate=0.5,
    )

    with pytest.raises(ValueError, match="increase strictly"):
        tracker.update(
            simulator_transitions=10,
            environment_return_mean=1.0,
            success_rate=0.5,
        )
