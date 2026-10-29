"""Read saved checkpoint evaluations without importing Isaac or a learner."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

TRAINING_INTERACTIONS = "Samples/training_environment_interactions"
TOTAL_INTERACTIONS = "Samples/total_environment_interactions"


def evaluation_payload(result: dict[str, Any]) -> dict[str, float]:
    """Combine seeds: task reward / valid transitions, success / attempted episodes."""
    raw = json.loads(Path(result["evaluation_file"]).read_text())
    training_interactions = int(raw["training_environment_transitions"])
    if training_interactions != int(result["training_environment_transitions"]):
        raise ValueError("Evaluation and controller disagree on training interactions.")
    payload = {
        TRAINING_INTERACTIONS: training_interactions,
        TOTAL_INTERACTIONS: result["total_interactions"],
        "Samples/training_trajectories": raw["training_episodes"],
        "Samples/evaluation_interactions": result["cumulative_evaluation_transitions"],
        "Samples/evaluation_trajectories": result["cumulative_evaluation_episodes"],
    }
    if payload[TOTAL_INTERACTIONS] != (
        training_interactions + payload["Samples/evaluation_interactions"]
    ):
        raise ValueError(
            "Total interactions must include training and evaluation costs."
        )
    for mode in ("deterministic", "stochastic"):
        rows = [row for row in raw["rows"] if row["mode"] == mode]
        episodes = sum(row["episodes"] for row in rows)
        transitions = sum(row["retained_transitions"] for row in rows)
        if episodes <= 0 or transitions <= 0:
            raise ValueError(f"Missing completed {mode} evaluation episodes.")
        reward_sum = sum(row["return_mean"] * row["episodes"] for row in rows)
        prefix = f"Evaluation/{result['variant']}/{mode}"
        payload[f"{prefix}/reward_per_step"] = reward_sum / transitions
        payload[f"{prefix}/episode_return_mean"] = reward_sum / episodes
        payload[f"{prefix}/episodes"] = episodes
        for name, source in (
            ("sustained_success_rate", "held_goal_last25"),
            ("goal_reached_rate", "ever_goal_success"),
            ("lift_rate", "ever_lift_10cm"),
            ("early_termination_rate", "terminated_episodes"),
        ):
            successes = sum(row[source] for row in rows)
            if not 0 <= successes <= episodes:
                raise ValueError(f"Invalid {source} count: {successes}/{episodes}.")
            payload[f"{prefix}/{name}"] = successes / episodes
    if not all(math.isfinite(value) for value in payload.values()):
        raise ValueError("Evaluation metrics must be finite.")
    return payload


def checkpoint_events(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Group variants at a checkpoint; every event carries its own physical x-axis.

    Event ids include variants so a later transfer result can be appended at
    the same x coordinate without losing either variant's metrics.
    """
    grouped: dict[int, dict[str, Any]] = {}
    for result in sorted(
        job["evaluations"],
        key=lambda item: (
            item["training_environment_transitions"],
            item["total_interactions"],
        ),
    ):
        step = int(result["training_environment_transitions"])
        event = grouped.setdefault(step, {"variants": [], "payload": {}})
        variant = result["variant"]
        if variant in event["variants"]:
            raise ValueError(f"Duplicate {variant} evaluation at {step} interactions.")
        payload = evaluation_payload(result)
        event["variants"].append(variant)
        # A later transfer check must not move the earlier easy-task cost axis.
        payload[f"Samples/{variant}/total_environment_interactions"] = payload[
            TOTAL_INTERACTIONS
        ]
        event["payload"].update(payload)
    return [
        {
            "id": f"{step}:{','.join(sorted(event['variants']))}",
            "payload": event["payload"],
        }
        for step, event in sorted(grouped.items())
    ]
