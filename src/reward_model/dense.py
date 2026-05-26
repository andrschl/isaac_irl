from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from reward_model.base import BaseRewardModel
from reward_model.projections import (
    elastic_net_proximal,
    project_onto_l1_ball,
    project_onto_l2_ball,
    soft_threshold,
)


_REGULARIZATION_OPTIONS = frozenset({"none", "l1", "l2", "elastic"})
_LINEAR_PROJECTION_OPTIONS = frozenset({"none", "l1_ball", "l2_ball"})


@dataclass(slots=True)
class RewardModelCfg:
    """
    Feature-based reward model config.

    The reward model consumes features directly (not observations/actions).

    Dense model: hidden_dims, activation.
    Linear model: linear_projection, linear_projection_radius.
    Both: regularization, regularization_strength, elastic_alpha.
    """

    num_features: int
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    is_linear: bool = False
    activation: str = "elu"

    # Regularization (both dense and linear)
    regularization: str = "none"  # "none" | "l1" | "l2" | "elastic"
    regularization_strength: float = 0.0
    elastic_alpha: float = 0.5  # L1 fraction when regularization=="elastic"

    # Linear-only: projection onto norm balls (applied after optimizer.step)
    linear_projection: str = "none"  # "none" | "l1_ball" | "l2_ball"
    linear_projection_radius: float = 1.0


class _FeatureRewardModelBase(BaseRewardModel):
    """Common feature-reward model plumbing for [N, D] and [B, T, D] inputs."""

    def __init__(self, cfg: RewardModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._validate_common_cfg(cfg)

    @staticmethod
    def _validate_common_cfg(cfg: RewardModelCfg) -> None:
        if cfg.num_features <= 0:
            raise ValueError(f"`num_features` must be > 0, got {cfg.num_features}")
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

    def reset(self, dones: torch.Tensor | None = None) -> None:
        # Stateless reward model
        del dones

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

        if feats.shape[-1] != int(self.cfg.num_features):
            raise ValueError(
                f"Expected feature dimension D={self.cfg.num_features}, "
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

    def apply_proximal_step(self) -> None:
        """Apply L1 or elastic net proximal operator to all parameters."""
        cfg = self.cfg
        if (
            cfg.regularization not in ("l1", "elastic")
            or cfg.regularization_strength <= 0
        ):
            return
        strength = float(cfg.regularization_strength)
        if cfg.regularization == "l1":
            lam1, lam2 = strength, 0.0
        else:
            alpha = float(cfg.elastic_alpha)
            lam1 = alpha * strength
            lam2 = (1.0 - alpha) * strength
        with torch.no_grad():
            for param in self.parameters():
                if cfg.regularization == "l1":
                    param.data.copy_(soft_threshold(param.data, lam1))
                else:
                    param.data.copy_(elastic_net_proximal(param.data, lam1, lam2))


class DenseFeatureRewardModel(_FeatureRewardModelBase):
    """Dense MLP reward model over features.

    Supported input shapes:
      - feats [N, D] -> rewards [N]
      - feats [B, T, D], mask [B, T] -> rewards [B, T]
    """

    def __init__(self, cfg: RewardModelCfg) -> None:
        if cfg.is_linear:
            raise ValueError("DenseFeatureRewardModel requires cfg.is_linear=False.")
        if cfg.linear_projection != "none":
            raise ValueError(
                "`linear_projection` is only valid for LinearFeatureRewardModel. "
                f"Got linear_projection={cfg.linear_projection!r}."
            )
        super().__init__(cfg)
        hidden_dims = tuple(int(hidden_dim) for hidden_dim in cfg.hidden_dims)
        if len(hidden_dims) == 0:
            raise ValueError(
                "`hidden_dims` must be non-empty for DenseFeatureRewardModel."
            )
        self.reward = self._build_reward_network(cfg, hidden_dims)

    @staticmethod
    def _build_reward_network(
        cfg: RewardModelCfg, hidden_dims: tuple[int, ...]
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_dim = cfg.num_features
        act = get_activation(cfg.activation)

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError(f"Hidden dims must be positive, got {hidden_dims}")
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act.__class__())  # fresh activation instance per layer
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)


class LinearFeatureRewardModel(_FeatureRewardModelBase):
    """Bias-free linear reward model over features.

    Linearity lets discounted returns use the identity:
    sum_t gamma^t r(f_t) == r(sum_t gamma^t f_t).
    """

    def __init__(self, cfg: RewardModelCfg) -> None:
        if not cfg.is_linear:
            raise ValueError("LinearFeatureRewardModel requires cfg.is_linear=True.")
        super().__init__(cfg)
        # Linear reward stays bias-free by design.
        self.reward = nn.Linear(cfg.num_features, 1, bias=False)

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
        """Project linear weights onto the configured L1 or L2 ball."""
        cfg = self.cfg
        if cfg.linear_projection == "none":
            return
        radius = float(cfg.linear_projection_radius)
        weight = self.reward.weight.data
        with torch.no_grad():
            if cfg.linear_projection == "l2_ball":
                projected = project_onto_l2_ball(weight, radius)
            else:
                projected = project_onto_l1_ball(weight, radius)
            self.reward.weight.data.copy_(projected)


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "relu":
        return nn.ReLU()
    if name == "lrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Invalid activation function: {name}")
