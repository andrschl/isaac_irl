"""Versioned metadata describing what a recorded demonstration file means.

Demonstrations outlive the code that wrote them. These attributes are what lets
a loader tell "recorded the way this run needs" from "recorded some other way
and shaped the same", instead of reinterpreting an old file into silence.
"""

from __future__ import annotations

from typing import Any

FEATURE_SEMANTICS_VERSION = "isaac_reward_manager_pre_reset_v1"
FEATURE_CAPTURE_TIMING = "inside_reward_manager_before_auto_reset"
FEATURE_VALUE_SEMANTICS = "raw_unweighted_reward_terms"

TRANSITION_SEMANTICS_VERSION = "complete_episode_transitions_v1"
"""Every recorded step is one transition carrying its own pre-reset successor.

A ``v1`` file stores ``next_obs``, ``terminated`` and ``truncated`` per step and
keeps the final transition of every episode. Files written before this format
dropped that last transition and stored no termination information at all, so
their bootstrap mask could only be guessed - which is exactly the guess an
off-policy critic cannot survive. They are rejected, not upgraded.
"""


def demonstration_metadata(
    *,
    horizon: int,
    step_dt: float,
    is_finite_horizon: bool,
    action_bound: float | None,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """The attributes written on ``data/`` when demonstrations are recorded."""
    metadata: dict[str, Any] = {
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "feature_capture_timing": FEATURE_CAPTURE_TIMING,
        "feature_value_semantics": FEATURE_VALUE_SEMANTICS,
        "transition_semantics_version": TRANSITION_SEMANTICS_VERSION,
        "episode_horizon": int(horizon),
        "step_dt": float(step_dt),
        "is_finite_horizon": bool(is_finite_horizon),
        "feature_names": list(feature_names),
    }
    if action_bound is not None:
        metadata["action_bound"] = float(action_bound)
    return metadata


def require_transition_semantics(
    attributes: Any,
    *,
    path: str,
    horizon: int,
    step_dt: float,
    is_finite_horizon: bool,
) -> None:
    """Refuse demonstrations recorded on a different episode than this run.

    The horizon decides where an episode ends, ``step_dt`` decides what a unit
    of reward is worth, and the finite-horizon flag decides whether anything
    bootstraps past the end. A demonstration recorded under other answers is
    not this task's data, however well its tensors line up.
    """
    stored = attributes.get("transition_semantics_version")
    if stored != TRANSITION_SEMANTICS_VERSION:
        raise ValueError(
            f"Demonstrations at {path!r} declare transition semantics "
            f"{stored!r}, not {TRANSITION_SEMANTICS_VERSION!r}. Re-record them "
            "with scripts/record_demos.py; older files carry no terminal "
            "information and cannot be upgraded."
        )
    mismatches = {
        name: (expected, attributes.get(name))
        for name, expected, matches in (
            (
                "episode_horizon",
                horizon,
                int(attributes.get("episode_horizon", -1)) == int(horizon),
            ),
            (
                "step_dt",
                step_dt,
                abs(float(attributes.get("step_dt", 0.0)) - float(step_dt)) < 1e-9,
            ),
            (
                "is_finite_horizon",
                is_finite_horizon,
                bool(attributes.get("is_finite_horizon", False))
                == bool(is_finite_horizon),
            ),
        )
        if not matches
    }
    if mismatches:
        raise ValueError(
            f"Demonstrations at {path!r} were recorded on a different episode "
            f"than this run: {mismatches} (attribute: expected, stored)."
        )
