"""Small algorithm-independent metric contract for experiment dashboards."""

from __future__ import annotations

import math

_TRAINING_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Overview/critic_loss",
        ("DirectQ/td_loss", "SuccessorQ/td_loss", "PPO/value_loss"),
    ),
    ("Overview/actor_parameter_delta", ("Actor/parameter_delta",)),
    ("Overview/actor_gradient_norm", ("Actor/gradient_norm",)),
    ("Overview/policy_std", ("Actor/policy_std_mean", "RL/policy_std_mean")),
    ("Overview/pmd_actor_kl", ("PMD/actor_projection_kl",)),
    ("Overview/pmd_target_kl", ("PMD/candidate_target_kl",)),
    ("Overview/policy_updates", ("Data/actor_updates",)),
    ("Overview/timeout_fraction", ("Data/truncated_fraction",)),
    (
        "Overview/bootstrap_disabled_fraction",
        ("Data/bootstrap_disabled_fraction",),
    ),
    ("Overview/action_clipped_fraction", ("Action/clipped_fraction",)),
    (
        "Overview/action_saturation_fraction",
        ("Action/boundary_saturation_fraction",),
    ),
    (
        "Overview/reward_reconstruction_error",
        ("Reward/reconstruction_max_error",),
    ),
)


def training_overview_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Select stable cross-algorithm aliases from one training payload."""
    overview: dict[str, float] = {}
    for overview_name, source_names in _TRAINING_ALIASES:
        for source_name in source_names:
            if source_name in metrics:
                overview[overview_name] = float(metrics[source_name])
                break
    return overview


class SampleEfficiencyTracker:
    """Track horizon-normalized return statistics across validations."""

    def __init__(
        self,
        *,
        validation_steps: int,
        return_per_step_thresholds: tuple[float, ...] = (),
    ) -> None:
        if validation_steps <= 0:
            raise ValueError("validation_steps must be > 0.")
        self._validation_steps = int(validation_steps)
        self._thresholds = tuple(float(value) for value in return_per_step_thresholds)
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
    ) -> dict[str, float]:
        """Return overview metrics for one deterministic validation."""
        if simulator_transitions < 0:
            raise ValueError("simulator_transitions must be >= 0.")
        if not math.isfinite(environment_return_mean):
            raise ValueError("environment_return_mean must be finite.")
        if success_rate is not None and not 0.0 <= success_rate <= 1.0:
            raise ValueError("success_rate must be in [0, 1].")
        if (
            self._previous_step is not None
            and simulator_transitions <= self._previous_step
        ):
            raise ValueError(
                "Validation simulator transitions must increase strictly; got "
                f"{simulator_transitions} after {self._previous_step}."
            )

        return_per_step = environment_return_mean / self._validation_steps
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
        overview = {
            "Overview/return_mean": float(environment_return_mean),
            "Overview/return_per_step": float(return_per_step),
            "Overview/best_return_per_step": float(self._best_return_per_step),
            "Overview/return_per_step_auc": float(mean_auc),
            "Overview/evaluation_steps": float(self._validation_steps),
        }

        for threshold in self._thresholds:
            if return_per_step >= threshold and threshold not in self._threshold_steps:
                self._threshold_steps[threshold] = simulator_transitions
            if threshold in self._threshold_steps:
                overview[_threshold_metric_name(threshold)] = float(
                    self._threshold_steps[threshold]
                )

        if success_rate is not None:
            self._best_success_rate = max(self._best_success_rate, success_rate)
            overview["Overview/success_rate"] = float(success_rate)
            overview["Overview/best_success_rate"] = float(self._best_success_rate)
        return overview


def _threshold_metric_name(threshold: float) -> str:
    label = format(threshold, ".6g").replace("-", "minus_").replace(".", "_")
    return f"Overview/transitions_to_return_per_step_{label}"
