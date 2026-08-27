"""Small reward-scoring interfaces used by off-policy value backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from reward_model.base import BaseRewardModel
from reward_model.dense import LinearFeatureRewardModel


class RewardScorer(Protocol):
    """Score raw feature tensors [...,D] into scalar rewards [...]."""

    @property
    def feature_dim(self) -> int: ...

    def score(self, features: torch.Tensor) -> torch.Tensor: ...


class LinearRewardScorer(RewardScorer, Protocol):
    """Expose the current linear reward for successor scalarization."""

    feature_names: tuple[str, ...]

    @property
    def weights(self) -> torch.Tensor: ...

    @property
    def step_dt(self) -> torch.Tensor: ...


class LearnedRewardScorer:
    """Adapt a learned feature reward model to the off-policy scoring API."""

    def __init__(self, model: BaseRewardModel, *, feature_dim: int) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0.")
        self.model = model
        self._feature_dim = int(feature_dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def score(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected features [...,{self.feature_dim}], got {features.shape}."
            )
        rewards = self.model.get_reward_from_features(features)
        if rewards.shape != features.shape[:-1]:
            raise ValueError(
                "Learned reward output must match features.shape[:-1], got "
                f"{rewards.shape} and {features.shape[:-1]}."
            )
        return rewards


class LearnedLinearRewardScorer(LearnedRewardScorer):
    """Expose live learned linear weights without copying or caching them."""

    def __init__(
        self,
        model: LinearFeatureRewardModel,
        *,
        feature_names: Sequence[str],
        step_dt: float,
    ) -> None:
        names = tuple(str(name) for name in feature_names)
        if len(names) != model.cfg.num_features:
            raise ValueError(
                "feature_names must match the learned linear reward dimension."
            )
        if step_dt <= 0.0:
            raise ValueError("step_dt must be > 0.")
        super().__init__(model, feature_dim=model.cfg.num_features)
        self.feature_names = names
        self._step_dt = torch.tensor(float(step_dt), dtype=torch.float32)

    @property
    def weights(self) -> torch.Tensor:
        model = self.model
        if not isinstance(model, LinearFeatureRewardModel):
            raise TypeError("Expected a LinearFeatureRewardModel.")
        return model.reward.weight.squeeze(0)

    @property
    def step_dt(self) -> torch.Tensor:
        return self._step_dt.to(self.weights)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Score raw reward-manager features with explicit environment dt."""
        if features.ndim < 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected features [...,{self.feature_dim}], got {features.shape}."
            )
        weights = self.weights.to(device=features.device, dtype=features.dtype)
        step_dt = self.step_dt.to(device=features.device, dtype=features.dtype)
        return step_dt * torch.einsum("...d,d->...", features, weights)
