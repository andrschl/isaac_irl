"""The small cross-algorithm metric contract shown on the run dashboard.

Every training run emits `Summary/*` regardless of algorithm or reward source.
It answers one question - is this run working? - from four angles: task
performance, reward learning, policy learning, and behavioral cloning.
Detailed per-algorithm diagnostics keep their own namespaces (`PMD/*`,
`DirectQ/*`, `SuccessorQ/*`, `Data/*`, `Action/*`) and are not repeated here.
"""

from __future__ import annotations

import math

# Summary name <- the first source key that a payload happens to carry. The
# table is what keeps algorithms from having to know about the dashboard.
_TRAINING_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Summary/critic_loss", ("DirectQ/td_loss", "SuccessorQ/td_loss", "RL/value_loss")),
    ("Summary/q_mean", ("Value/q_mean",)),
    ("Summary/policy_std", ("Actor/policy_std_mean", "RL/policy_std_mean")),
    ("Summary/actor_kl", ("PMD/actor_projection_kl",)),
    ("Summary/policy_updates", ("Data/actor_updates",)),
    ("Summary/train_reward_per_step", ("Train/reward_per_step",)),
    ("Summary/train_episode_return", ("Train/episode_return_mean",)),
    ("Summary/train_episode_success_rate", ("Train/episode_success_rate",)),
    ("Summary/reward_loss", ("IRL/reward_loss",)),
    ("Summary/feature_gap_norm", ("IRL/feature_exp_diff_norm",)),
    ("Summary/reward_weight_norm", ("IRL/reward_param_norm",)),
    ("Summary/reward_weight_cosine", ("IRL/reward_weight_cosine",)),
    ("Summary/bc_loss", ("BC/loss", "Actor/bc_loss")),
    ("Summary/bc_val_loss", ("BC/val_loss",)),
)

TRANSITIONS = "Summary/transitions"
"""Training transitions retained for learning (excludes evaluation/overhead)."""


def training_summary(metrics: dict[str, float]) -> dict[str, float]:
    """Select the stable summary aliases present in one training payload."""
    summary: dict[str, float] = {}
    for summary_name, source_names in _TRAINING_ALIASES:
        for source_name in source_names:
            if source_name in metrics:
                summary[summary_name] = float(metrics[source_name])
                break
    return summary


class SampleEfficiencyTracker:
    """Turn repeated validations into best/area/time-to-threshold numbers."""

    def __init__(
        self,
        *,
        validation_steps: int,
        return_per_step_thresholds: tuple[float, ...] = (),
        success_rate_thresholds: tuple[float, ...] = (),
    ) -> None:
        if validation_steps <= 0:
            raise ValueError(f"validation_steps must be > 0, got {validation_steps}.")
        self._validation_steps = int(validation_steps)
        self._thresholds = tuple(float(value) for value in return_per_step_thresholds)
        self._success_thresholds = tuple(
            float(value) for value in success_rate_thresholds
        )
        self._success_threshold_steps: dict[float, int] = {}
        self._first_step: int | None = None
        self._previous_step: int | None = None
        self._previous_return_per_step: float | None = None
        self._return_auc = 0.0
        self._best_return_per_step = -math.inf
        self._best_success_rate = -math.inf
        self._threshold_steps: dict[float, int] = {}

    def update(
        self,
        *,
        simulator_transitions: int,
        environment_return_mean: float,
        success_rate: float | None,
        environment_return_per_step: float | None = None,
    ) -> dict[str, float]:
        """Return summary metrics for one deterministic validation."""
        if simulator_transitions < 0:
            raise ValueError(
                f"simulator_transitions must be >= 0, got {simulator_transitions}."
            )
        if not math.isfinite(environment_return_mean):
            raise ValueError(
                f"environment_return_mean must be finite, got {environment_return_mean}."
            )
        if success_rate is not None and not 0.0 <= success_rate <= 1.0:
            raise ValueError(f"success_rate must be in [0, 1], got {success_rate}.")
        if (
            self._previous_step is not None
            and simulator_transitions <= self._previous_step
        ):
            raise ValueError(
                "Validation simulator transitions must increase strictly; got "
                f"{simulator_transitions} after {self._previous_step}."
            )

        return_per_step = (
            environment_return_mean / self._validation_steps
            if environment_return_per_step is None
            else environment_return_per_step
        )
        if not math.isfinite(return_per_step):
            raise ValueError("environment_return_per_step must be finite.")
        if self._first_step is None:
            self._first_step = simulator_transitions
        if self._previous_step is not None:
            transition_delta = simulator_transitions - self._previous_step
            self._return_auc += (
                transition_delta
                * (return_per_step + float(self._previous_return_per_step))
                / 2.0
            )
        self._previous_step = simulator_transitions
        self._previous_return_per_step = return_per_step
        self._best_return_per_step = max(self._best_return_per_step, return_per_step)

        elapsed_transitions = simulator_transitions - self._first_step
        mean_auc = (
            self._return_auc / elapsed_transitions if elapsed_transitions else 0.0
        )
        summary = {
            "Summary/return_per_step": float(return_per_step),
            "Summary/best_return_per_step": float(self._best_return_per_step),
            "Summary/return_per_step_auc": float(mean_auc),
        }

        for threshold in self._thresholds:
            if return_per_step >= threshold and threshold not in self._threshold_steps:
                self._threshold_steps[threshold] = simulator_transitions
            if threshold in self._threshold_steps:
                summary[_threshold_metric_name(threshold)] = float(
                    self._threshold_steps[threshold]
                )

        if success_rate is not None:
            self._best_success_rate = max(self._best_success_rate, success_rate)
            summary["Summary/success_rate"] = float(success_rate)
            summary["Summary/best_success_rate"] = float(self._best_success_rate)
            for threshold in self._success_thresholds:
                if (
                    success_rate >= threshold
                    and threshold not in self._success_threshold_steps
                ):
                    self._success_threshold_steps[threshold] = simulator_transitions
                if threshold in self._success_threshold_steps:
                    summary[_success_metric_name(threshold)] = float(
                        self._success_threshold_steps[threshold]
                    )
        return summary


def _success_metric_name(threshold: float) -> str:
    return f"Summary/transitions_to_success_rate_{_threshold_label(threshold)}"


def _threshold_label(threshold: float) -> str:
    return format(threshold, ".6g").replace("-", "minus_").replace(".", "_")


def _threshold_metric_name(threshold: float) -> str:
    return f"Summary/transitions_to_return_per_step_{_threshold_label(threshold)}"
