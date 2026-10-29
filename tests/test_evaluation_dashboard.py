"""Regression checks for fair, interaction-indexed checkpoint comparisons."""

import json

import pytest

from data.checkpoint_evaluations import (
    TOTAL_INTERACTIONS,
    TRAINING_INTERACTIONS,
    checkpoint_events,
    evaluation_payload,
)


def _result(tmp_path, variant="lift_easy", cost=16000):
    rows = []
    for mode in ("deterministic", "stochastic"):
        for count, transitions, reward in ((2, 500, 5.0), (1, 100, 20.0)):
            rows.append(
                {
                    "mode": mode,
                    "episodes": count,
                    "retained_transitions": transitions,
                    "return_mean": reward,
                    "held_goal_last25": 1,
                    "ever_goal_success": 1,
                    "ever_lift_10cm": 1,
                    "terminated_episodes": 0,
                }
            )
    path = tmp_path / f"{variant}.json"
    path.write_text(
        json.dumps(
            {
                "training_environment_transitions": 8000,
                "training_transitions": 7900,
                "training_episodes": 32,
                "rows": rows,
            }
        )
    )
    return {
        "evaluation_file": str(path),
        "variant": variant,
        "training_environment_transitions": 8000,
        "cumulative_evaluation_transitions": cost,
        "cumulative_evaluation_episodes": 6,
        "total_interactions": 8000 + cost,
    }


def test_metrics_use_physical_steps_and_weighted_episode_lengths(tmp_path):
    payload = evaluation_payload(_result(tmp_path))
    assert payload[TRAINING_INTERACTIONS] == 8000  # neither iteration nor replay count
    assert payload[TOTAL_INTERACTIONS] == 24000
    prefix = "Evaluation/lift_easy/deterministic"
    assert payload[f"{prefix}/reward_per_step"] == pytest.approx(30 / 600)
    assert payload[f"{prefix}/episode_return_mean"] == 10
    assert payload[f"{prefix}/sustained_success_rate"] == pytest.approx(2 / 3)
    assert payload[f"{prefix}/early_termination_rate"] == 0


def test_transfer_at_same_checkpoint_preserves_easy_cost_axis(tmp_path):
    easy = _result(tmp_path)
    transfer = _result(tmp_path, "standard", cost=32000)
    events = checkpoint_events({"evaluations": [transfer, easy]})
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload[TRAINING_INTERACTIONS] == 8000
    assert payload["Samples/lift_easy/total_environment_interactions"] == 24000
    assert payload["Samples/standard/total_environment_interactions"] == 40000
    assert "Evaluation/lift_easy/deterministic/reward_per_step" in payload
    assert "Evaluation/standard/deterministic/reward_per_step" in payload
    assert events[0]["id"] != checkpoint_events({"evaluations": [easy]})[0]["id"]


def test_duplicate_evaluation_is_rejected(tmp_path):
    result = _result(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        checkpoint_events({"evaluations": [result, result]})


def test_conflicting_interaction_counts_are_rejected(tmp_path):
    result = _result(tmp_path)
    result["training_environment_transitions"] = 7900
    with pytest.raises(ValueError, match="disagree"):
        evaluation_payload(result)


def test_report_overlays_four_runs_with_physical_axes():
    wr = pytest.importorskip("wandb_workspaces.reports.v2")
    from training.evaluation_dashboard import build_report_blocks

    state = {
        "jobs": [
            {
                "evaluations": [
                    {"variant": "lift_easy", "training_environment_transitions": budget}
                ]
            }
            for budget in (147456000, 400000, 400000, 400000)
        ]
    }
    manifest = {
        "runs": {name: {"run_id": name} for name in ("ppo", "q25", "q50", "q75")}
    }
    grids = [
        block
        for block in build_report_blocks(state, manifest)
        if isinstance(block, wr.PanelGrid)
    ]
    assert len(grids) == 3
    for grid in grids:
        serialized_filters = grid.runsets[0]._to_model().filters["filters"]
        assert all(
            item["key"] == {"section": "run", "name": "name"}
            for item in serialized_filters
        )  # ids, not display names
        for name in manifest["runs"]:
            assert name in grid.runsets[0].filters
        for panel in grid.panels:
            assert panel.x.startswith("Samples/")
            assert panel.max_runs_to_show == 4
            assert panel.smoothing_type == "none"
            assert not panel.aggregate
    assert all(
        panel.range_x == (0, 400000) and not panel.log_x for panel in grids[0].panels
    )
    assert all(
        panel.x == TRAINING_INTERACTIONS and panel.log_x for panel in grids[1].panels
    )


def test_queued_trials_do_not_hide_equal_budget_or_failure_plots():
    wr = pytest.importorskip("wandb_workspaces.reports.v2")
    from training.evaluation_dashboard import build_report_blocks

    state = {
        "protocol": {"include_failure_diagnostics": True},
        "jobs": [
            {
                "evaluations": [
                    {"variant": "lift_easy", "training_environment_transitions": 800000}
                ]
            },
            {
                "evaluations": [
                    {"variant": "lift_easy", "training_environment_transitions": 400000}
                ]
            },
            {"evaluations": []},
        ],
    }
    manifest = {"runs": {name: {"run_id": name} for name in ("ppo", "ready", "queued")}}
    grids = [
        block
        for block in build_report_blocks(state, manifest)
        if isinstance(block, wr.PanelGrid)
    ]
    assert len(grids) == 3
    assert len(grids[0].panels) == 8
    assert all(panel.range_x == (0, 400000) for panel in grids[0].panels)
    assert sum("early_termination_rate" in panel.y[0] for panel in grids[0].panels) == 2
