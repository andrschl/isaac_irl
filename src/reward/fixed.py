"""Immutable linear feature reward used by fixed-reward RL experiments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class FixedLinearReward(nn.Module):
    """Score raw features with immutable environment reward weights.

    Shapes:
        weights: [D]
        features: [..., D]
        rewards: [...]
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        weights: torch.Tensor,
        *,
        step_dt: float,
    ) -> None:
        super().__init__()
        names = tuple(str(name) for name in feature_names)
        weight_tensor = torch.as_tensor(weights, dtype=torch.float32)
        if weight_tensor.ndim != 1 or weight_tensor.numel() != len(names):
            raise ValueError(
                f"Expected weights [{len(names)}], got {tuple(weight_tensor.shape)}."
            )
        if not names or len(set(names)) != len(names):
            raise ValueError("feature_names must be non-empty and unique.")
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be > 0, got {step_dt}.")
        if not torch.isfinite(weight_tensor).all():
            raise ValueError("Fixed reward weights must be finite.")
        self.feature_names = names
        self.register_buffer("weights", weight_tensor.clone())
        self.register_buffer(
            "step_dt", torch.tensor(float(step_dt), dtype=torch.float32)
        )

    @property
    def feature_dim(self) -> int:
        return int(self.weights.numel())

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Return rewards [...] for raw features [...,D]."""
        if features.ndim < 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected features [...,{self.feature_dim}], got {tuple(features.shape)}."
            )
        if not torch.isfinite(features).all():
            raise ValueError("Cannot score non-finite fixed-reward features.")
        weights = self.weights.to(device=features.device, dtype=features.dtype)
        step_dt = self.step_dt.to(device=features.device, dtype=features.dtype)
        return torch.einsum("...d,d->...", features, weights) * step_dt

    def reconstruction_error(
        self, features: torch.Tensor, environment_rewards: torch.Tensor
    ) -> torch.Tensor:
        """Return per-sample absolute error with shape features.shape[:-1]."""
        predicted = self.score(features)
        if predicted.shape != environment_rewards.shape:
            raise ValueError(
                "Environment rewards must match fixed reward output shape, "
                f"got {tuple(environment_rewards.shape)} and {tuple(predicted.shape)}."
            )
        return (predicted - environment_rewards.to(predicted)).abs()
