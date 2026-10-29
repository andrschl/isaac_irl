"""Small reward-scoring interfaces used by off-policy value backends.

One unit convention runs through every linear scorer here: features are the
*raw, unweighted* reward-manager terms, and a scorer returns the reward for one
environment step, so it multiplies by the environment's ``step_dt``. The IRL
feature expectations are accumulated the same way, which is what lets a learned
weight vector be read against the environment's own reward weights.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import torch

from reward.base import BaseRewardModel
from reward.fixed import FixedLinearReward
from reward.model import LinearFeatureRewardModel


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
    """Adapt a learned feature reward model to the off-policy scoring API.

    Unlike the linear scorers, this one hands back the model's output as it is.
    A dense reward model is free to represent whatever scale it likes, so there
    is no per-step quantity to integrate with ``step_dt``; its output already
    *is* the per-step reward. Keep that in mind when comparing a dense run's
    returns against a linear one's.
    """

    def __init__(self, model: BaseRewardModel, *, feature_dim: int) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0.")
        self.model = model
        self._feature_dim = int(feature_dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def score(self, features: torch.Tensor) -> torch.Tensor:
        _require_feature_shape(features, self.feature_dim)
        rewards = self.model.get_reward_from_features(features)
        if rewards.shape != features.shape[:-1]:
            raise ValueError(
                "Learned reward output must match features.shape[:-1], got "
                f"{rewards.shape} and {features.shape[:-1]}."
            )
        return rewards


class LearnedLinearRewardScorer(LearnedRewardScorer):
    """Expose live coefficients on raw features, including configured scales."""

    def __init__(
        self,
        model: LinearFeatureRewardModel,
        *,
        feature_names: Sequence[str],
        step_dt: float,
    ) -> None:
        names = tuple(str(name) for name in feature_names)
        if len(names) != model.num_features:
            raise ValueError(
                "feature_names must match the learned linear reward dimension."
            )
        if step_dt <= 0.0:
            raise ValueError("step_dt must be > 0.")
        super().__init__(model, feature_dim=model.num_features)
        self.feature_names = names
        self._step_dt = torch.tensor(float(step_dt), dtype=torch.float32)

    @property
    def weights(self) -> torch.Tensor:
        """Live learned coefficients [D]; never a snapshot."""
        model = self.model
        if not isinstance(model, LinearFeatureRewardModel):
            raise TypeError("Expected a LinearFeatureRewardModel.")
        return model.effective_weights

    @property
    def step_dt(self) -> torch.Tensor:
        return self._step_dt.to(self.weights)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Score raw reward-manager features with explicit environment dt."""
        _require_feature_shape(features, self.feature_dim)
        weights = self.weights.to(device=features.device, dtype=features.dtype)
        step_dt = self.step_dt.to(device=features.device, dtype=features.dtype)
        return step_dt * torch.einsum("...d,d->...", features, weights)


class AdditiveLinearRewardScorer:
    """A fixed reward plus a learned one, over a shared named feature basis.

    The fixed part is whatever the experiment refuses to learn - on the sparse
    arms, the one terminal-success coefficient - and stays exactly where it was
    set. The learned part covers a *subset* of the same basis, so the sparse
    coordinate cannot be moved by the reward learner even though the critic
    sees one feature vector. The combined weights are recomputed on every read
    rather than cached, so a reward update is visible to the next scoring call.

    Shapes: ``weights [D]``, ``features [..., D]``, ``rewards [...]``.
    """

    def __init__(
        self,
        fixed: FixedLinearReward,
        learned: LearnedLinearRewardScorer,
        *,
        fixed_coefficient: float = 1.0,
        learned_coefficient: float = 1.0,
    ) -> None:
        coefficients = (fixed_coefficient, learned_coefficient)
        if any(not math.isfinite(value) or value < 0 for value in coefficients):
            raise ValueError("Reward mixture weights must be finite and nonnegative.")
        if sum(coefficients) == 0:
            raise ValueError("At least one reward mixture weight must be positive.")
        self.fixed_coefficient = float(fixed_coefficient)
        self.learned_coefficient = float(learned_coefficient)
        missing = [
            name for name in learned.feature_names if name not in fixed.feature_names
        ]
        if missing:
            raise ValueError(
                f"Learned reward features {missing} are not in the fixed reward "
                f"basis {list(fixed.feature_names)}."
            )
        if not torch.allclose(fixed.step_dt.cpu(), learned.step_dt.cpu()):
            raise ValueError(
                "Fixed and learned reward must use the same step_dt, got "
                f"{float(fixed.step_dt)} and {float(learned.step_dt)}."
            )
        self.fixed = fixed
        self.learned = learned
        self.feature_names = fixed.feature_names
        # Column of the shared basis each learned coefficient belongs to. [L]
        self._learned_columns = torch.tensor(
            [self.feature_names.index(name) for name in learned.feature_names],
            dtype=torch.long,
        )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def step_dt(self) -> torch.Tensor:
        return self.learned.step_dt

    @property
    def weights(self) -> torch.Tensor:
        """Combined coefficients [D], scattered back onto the shared basis."""
        learned = self.learned.weights
        columns = self._learned_columns.to(learned.device)
        fixed = self.fixed_coefficient * self.fixed.weights.to(learned)
        return fixed.index_add(0, columns, self.learned_coefficient * learned)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        _require_feature_shape(features, self.feature_dim)
        weights = self.weights.to(device=features.device, dtype=features.dtype)
        step_dt = self.step_dt.to(device=features.device, dtype=features.dtype)
        return step_dt * torch.einsum("...d,d->...", features, weights)


def _require_feature_shape(features: torch.Tensor, feature_dim: int) -> None:
    if features.ndim < 2 or features.shape[-1] != feature_dim:
        raise ValueError(
            f"Expected features [...,{feature_dim}], got {tuple(features.shape)}."
        )
