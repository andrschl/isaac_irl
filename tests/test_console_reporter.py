"""Tests for runner.console_reporter."""

from __future__ import annotations

from runner.console_reporter import ConsoleReporter


def test_start_announces_iteration_count(capsys):
    ConsoleReporter().start(num_iterations=42)
    out = capsys.readouterr().out
    assert "42" in out
    assert "Starting learning" in out


def test_iteration_summary_formats_core_metrics(capsys):
    ConsoleReporter().iteration_summary(
        iteration=4,
        rl_metrics={"RL/policy_loss": 0.123},
        reward_metrics={
            "IRL/reward_loss": 0.456,
            "IRL/feature_exp_diff_norm": 0.789,
        },
    )
    out = capsys.readouterr().out
    assert "iter 5" in out  # iteration+1
    assert "policy_loss=0.1230" in out
    assert "reward_loss=0.4560" in out
    assert "feat_diff_norm=0.7890" in out
    assert "bc_loss" not in out  # no BC payload provided


def test_iteration_summary_includes_bc_loss_when_present(capsys):
    ConsoleReporter().iteration_summary(
        iteration=0,
        rl_metrics={"RL/policy_loss": 0.0, "BC/loss": 0.25},
        reward_metrics={"IRL/reward_loss": 0.0, "IRL/feature_exp_diff_norm": 0.0},
    )
    out = capsys.readouterr().out
    assert "bc_loss=0.2500" in out


def test_validation_summary_includes_label_and_strips_metric_namespace(capsys):
    ConsoleReporter().validation_summary(
        iteration=9,
        metrics={"Validation/stochastic/success_rate": 0.75},
        label="mode=stochastic",
    )
    out = capsys.readouterr().out
    assert "iter 10" in out
    assert "mode=stochastic" in out
    assert "success_rate=0.7500" in out


def test_validation_summary_skips_empty_metrics(capsys):
    ConsoleReporter().validation_summary(iteration=0, metrics={})
    assert capsys.readouterr().out == ""


def test_validation_features_filters_by_prefix_and_strips_it(capsys):
    payload = {
        "ValidationFeatures/expert/reach": 1.0,
        "ValidationFeatures/expert/lift": 2.0,
        "Validation/expert/success_rate": 0.5,  # different prefix, should be excluded
    }
    ConsoleReporter().validation_features(
        iteration=2,
        label="group=expert",
        prefix="ValidationFeatures/expert/",
        payload=payload,
    )
    out = capsys.readouterr().out
    assert "iter 3" in out
    assert "group=expert" in out
    assert "reach=1.0000" in out
    assert "lift=2.0000" in out
    assert "success_rate" not in out  # filtered out by prefix


def test_validation_features_skips_when_no_matching_keys(capsys):
    ConsoleReporter().validation_features(
        iteration=0,
        label="group=imitator",
        prefix="ValidationFeatures/imitator/",
        payload={"OtherPrefix/x": 1.0},
    )
    assert capsys.readouterr().out == ""
