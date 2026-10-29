from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from einops import rearrange

from config.schema import RewardModelCfg
from reward.base import BaseRewardModel
from reward.projections import project_onto_l2_ball

_REGULARIZATION_OPTIONS = frozenset({"none", "l2"})
_LINEAR_PROJECTION_OPTIONS = frozenset({"none", "l2_ball"})


class _FeatureRewardModelBase(BaseRewardModel):
    """Common feature-reward model plumbing for [N, D] and [B, T, D] inputs."""

    def __init__(self, cfg: RewardModelCfg, *, num_features: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_features = int(num_features)
        if self.num_features <= 0:
            raise ValueError(f"`num_features` must be > 0, got {num_features}.")
        self._validate_common_cfg(cfg)

    @staticmethod
    def _validate_common_cfg(cfg: RewardModelCfg) -> None:
        if cfg.regularization not in _REGULARIZATION_OPTIONS:
            raise ValueError(
                f"`regularization` must be one of {sorted(_REGULARIZATION_OPTIONS)}, "
                f"got {cfg.regularization!r}."
            )
        if cfg.linear_projection not in _LINEAR_PROJECTION_OPTIONS:
            raise ValueError(
                f"`linear_projection` must be one of {sorted(_LINEAR_PROJECTION_OPTIONS)}, "
                f"got {cfg.linear_projection!r}."
            )
        if cfg.linear_projection_radius <= 0:
            raise ValueError(
                f"`linear_projection_radius` must be > 0, got {cfg.linear_projection_radius}."
            )

    def forward(
        self, feats: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Alias for compatibility with code calling ``reward(feats, mask)``."""
        return self.get_reward_from_features(feats, mask)

    def get_reward_from_features(
        self,
        feats: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-step rewards from features.

        Args:
            feats: [N, D] or [B, T, D]
            mask: Optional boolean mask matching output shape ([N] or [B, T]).

        Returns:
            rewards with shape feats.shape[:-1]
        """
        if not isinstance(feats, torch.Tensor):
            feats = torch.as_tensor(feats)

        if feats.ndim not in (2, 3):
            raise ValueError(
                f"`feats` must be [N,D] or [B,T,D], got {tuple(feats.shape)}"
            )

        if feats.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected feature dimension D={self.num_features}, "
                f"got feats shape {tuple(feats.shape)}."
            )

        if feats.ndim == 2:
            # feats: [N, D] -> rewards: [N]
            flat_rewards = self.reward(feats)
            rewards = rearrange(flat_rewards, "n 1 -> n")
        else:
            # feats: [B, T, D] -> rewards: [B, T]
            batch_size, traj_len, _ = feats.shape
            flat_feats = rearrange(feats, "b t d -> (b t) d")
            flat_rewards = self.reward(flat_feats)
            rewards = rearrange(
                flat_rewards, "(b t) 1 -> b t", b=batch_size, t=traj_len
            )

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, device=rewards.device)
            if tuple(mask.shape) != tuple(rewards.shape):
                raise ValueError(
                    "Expected mask shape to match reward shape, "
                    f"got mask {tuple(mask.shape)} and rewards {tuple(rewards.shape)}."
                )
            rewards = rewards * mask.to(dtype=rewards.dtype, device=rewards.device)

        return rewards

    def get_regularization_loss(self) -> torch.Tensor:
        """Differentiable L2 regularization penalty. Returns 0 when disabled."""
        cfg = self.cfg
        if cfg.regularization != "l2" or cfg.regularization_strength <= 0:
            return super().get_regularization_loss()
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.parameters():
            total = total + param.pow(2).sum()
        return cfg.regularization_strength * total


class DenseFeatureRewardModel(_FeatureRewardModelBase):
    """Dense MLP reward model over features.

    Supported input shapes:
      - feats [N, D] -> rewards [N]
      - feats [B, T, D], mask [B, T] -> rewards [B, T]
    """

    def __init__(self, cfg: RewardModelCfg, *, num_features: int) -> None:
        if cfg.is_linear:
            raise ValueError("DenseFeatureRewardModel requires cfg.is_linear=False.")
        if cfg.linear_projection != "none":
            raise ValueError(
                "`linear_projection` is only valid for LinearFeatureRewardModel. "
                f"Got linear_projection={cfg.linear_projection!r}."
            )
        super().__init__(cfg, num_features=num_features)
        hidden_dims = tuple(int(hidden_dim) for hidden_dim in cfg.hidden_dims)
        if len(hidden_dims) == 0:
            raise ValueError(
                "`hidden_dims` must be non-empty for DenseFeatureRewardModel."
            )
        self.reward = self._build_reward_network(
            cfg, hidden_dims, num_features=self.num_features
        )

    @staticmethod
    def _build_reward_network(
        cfg: RewardModelCfg, hidden_dims: tuple[int, ...], *, num_features: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = num_features
        make_activation = activation_factory(cfg.activation)

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError(f"Hidden dims must be positive, got {hidden_dims}")
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(make_activation())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)


class LinearFeatureRewardModel(_FeatureRewardModelBase):
    """Bias-free linear reward model over features.

    Linearity lets discounted returns use the identity:
    sum_t gamma^t r(f_t) == r(sum_t gamma^t f_t).
    """

    def __init__(
        self,
        cfg: RewardModelCfg,
        *,
        num_features: int,
        feature_scales: Sequence[float] | None = None,
    ) -> None:
        if not cfg.is_linear:
            raise ValueError("LinearFeatureRewardModel requires cfg.is_linear=True.")
        super().__init__(cfg, num_features=num_features)
        # Linear reward stays bias-free by design.
        self.reward = nn.Linear(self.num_features, 1, bias=False)
        nn.init.zeros_(self.reward.weight)
        scales = torch.as_tensor(
            [1.0] * self.num_features if feature_scales is None else feature_scales,
            dtype=torch.float32,
        )
        if (
            scales.shape != (self.num_features,)
            or not torch.isfinite(scales).all()
            or (scales <= 0).any()
        ):
            raise ValueError(
                f"Expected finite positive feature_scales [{self.num_features}], "
                f"got {scales.tolist()}."
            )
        # Scales are experiment configuration, not mutable learned state.
        self.register_buffer("feature_scales", scales, persistent=False)

    @property
    def effective_weights(self) -> torch.Tensor:
        """Raw-feature coefficients [D] used by the critic and scorer."""
        return self.reward.weight.squeeze(0) / self.feature_scales

    def get_reward_from_features(
        self, feats: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Score raw features [...,D] with weights on feats / feature_scales."""
        if not isinstance(feats, torch.Tensor):
            feats = torch.as_tensor(feats)
        if feats.ndim in (2, 3) and feats.shape[-1] == self.num_features:
            compute_dtype = feats.dtype if feats.is_floating_point() else torch.float32
            scales = self.feature_scales.to(device=feats.device, dtype=compute_dtype)
            feats = feats.to(dtype=compute_dtype) / scales
        return super().get_reward_from_features(feats, mask)

    def set_named_weights(
        self,
        feature_names: tuple[str, ...] | list[str],
        weights_by_name: dict[str, float],
    ) -> None:
        """Set the [1,D] reward weight from an explicitly ordered feature list."""
        if len(feature_names) != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} feature names, got {len(feature_names)}."
            )
        missing = [name for name in feature_names if name not in weights_by_name]
        if missing:
            raise ValueError(
                f"Ground-truth reward weights are missing feature names: {missing}."
            )
        weight = (
            torch.tensor(
                [weights_by_name[name] for name in feature_names],
                device=self.reward.weight.device,
                dtype=self.reward.weight.dtype,
            )
            * self.feature_scales
        )
        with torch.no_grad():
            self.reward.weight.copy_(weight.unsqueeze(0))

    def discounted_returns_from_features(
        self,
        feats: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Evaluate discounted returns via discounted feature sums.

        Args:
            feats: [B, T, D]
            mask:  [B, T]
            gamma: discount factor

        Returns:
            discounted returns [B]
        """
        self._validate_discount_inputs(feats, mask)
        model_device, model_dtype = self._parameter_device_dtype(fallback=feats)
        feats = feats.to(device=model_device, dtype=model_dtype)
        mask = mask.to(device=model_device, dtype=torch.bool)
        time_steps = feats.shape[1]
        powers = gamma ** torch.arange(
            time_steps, device=feats.device, dtype=feats.dtype
        )
        discounted_feats = torch.einsum(
            "btd,bt,t->bd", feats, mask.to(dtype=feats.dtype), powers
        )
        returns = self.get_reward_from_features(discounted_feats)  # [B]
        if not isinstance(returns, torch.Tensor):
            returns = torch.as_tensor(returns, device=model_device, dtype=model_dtype)
        if returns.ndim == 2 and returns.shape[-1] == 1:
            returns = returns.squeeze(-1)
        if returns.ndim != 1 or returns.shape[0] != feats.shape[0]:
            raise ValueError(
                "Expected linear reward returns with shape [B], "
                f"got {tuple(returns.shape)} for feats shape {tuple(feats.shape)}."
            )
        return returns

    def project_weights(self) -> None:
        """Project linear weights back onto the configured L2 ball."""
        cfg = self.cfg
        if cfg.linear_projection == "none":
            return
        with torch.no_grad():
            self.reward.weight.data.copy_(
                project_onto_l2_ball(
                    self.reward.weight.data, float(cfg.linear_projection_radius)
                )
            )


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "elu": nn.ELU,
    "selu": nn.SELU,
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def activation_factory(name: str) -> type[nn.Module]:
    """Return the activation class, so every layer builds its own instance."""
    activation = _ACTIVATIONS.get(name.lower())
    if activation is None:
        raise ValueError(
            f"Unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}."
        )
    return activation
