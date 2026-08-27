"""Configuration for fixed-reward RL evaluation and reporting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FixedRewardRunnerCfg:
    """Runner schedule and horizon-independent reporting choices."""

    save_interval: int = 50
    validation_interval: int = 1
    validation_steps: int = 250
    validate_before_training: bool = False
    validation_interval_transitions: int | None = None
    return_per_step_thresholds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.save_interval <= 0:
            raise ValueError("save_interval must be > 0.")
        if self.validation_interval < 0 or self.validation_steps <= 0:
            raise ValueError(
                "validation_interval must be >= 0 and validation_steps > 0."
            )
        if (
            self.validation_interval_transitions is not None
            and self.validation_interval_transitions <= 0
        ):
            raise ValueError("validation_interval_transitions must be > 0.")
        if any(
            not float("-inf") < value < float("inf")
            for value in self.return_per_step_thresholds
        ):
            raise ValueError("return_per_step_thresholds must contain finite values.")
        if len(set(self.return_per_step_thresholds)) != len(
            self.return_per_step_thresholds
        ):
            raise ValueError("return_per_step_thresholds must be unique.")
