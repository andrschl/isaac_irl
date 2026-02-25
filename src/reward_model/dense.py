from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

from reward_model.base import BaseRewardModel


@dataclass(slots=True)
class RewardModelCfg:
    """
    Feature-based reward model config.

    The reward model consumes features directly (not observations/actions).
    """
    num_features: int
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    is_linear: bool = False
    activation: str = "elu"


class RewardModel(BaseRewardModel):
    """
    Reward model over features.

    Supported input shapes:
      - [N, D]     -> returns [N]
      - [B, T, D]  -> returns [B, T]

    `mask` is optional and used only for shape validation / optional zeroing.
    """

    def __init__(self, cfg: RewardModelCfg):
        super().__init__()
        self.cfg = cfg

        if cfg.num_features <= 0:
            raise ValueError(f"`num_features` must be > 0, got {cfg.num_features}")

        self.reward = self._build_reward_network(cfg)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_reward_network(self, cfg: RewardModelCfg) -> nn.Module:
        if cfg.is_linear:
            # Linear reward stays bias-free by design.
            return nn.Linear(cfg.num_features, 1, bias=False)

        hidden_dims = tuple(int(h) for h in cfg.hidden_dims)
        if len(hidden_dims) == 0:
            raise ValueError("`hidden_dims` must be non-empty for non-linear reward model.")

        layers: list[nn.Module] = []
        in_dim = cfg.num_features
        act = get_activation(cfg.activation)

        for h in hidden_dims:
            if h <= 0:
                raise ValueError(f"Hidden dims must be positive, got {hidden_dims}")
            layers.append(nn.Linear(in_dim, h))
            layers.append(act.__class__())  # fresh instance per layer
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, dones: torch.Tensor | None = None) -> None:
        # Stateless reward model
        del dones

    @property
    def is_linear(self) -> bool:
        return bool(self.cfg.is_linear)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Alias for compatibility with code calling `reward(feats, mask)`.
        """
        return self.get_reward_from_features(feats, mask)

    def get_reward_from_features(
        self,
        feats: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute per-step rewards from features.

        Args:
            feats:
                [N, D] or [B, T, D]
            mask:
                Optional boolean mask matching output shape ([N] or [B, T]).
                Used for validation and zeroing padded outputs.

        Returns:
            rewards with shape feats.shape[:-1]
        """
        if not isinstance(feats, torch.Tensor):
            feats = torch.as_tensor(feats)

        if feats.ndim not in (2, 3):
            raise ValueError(f"`feats` must be [N,D] or [B,T,D], got {tuple(feats.shape)}")

        if feats.ndim == 2:
            # feats: [N, D] -> rewards: [N]
            flat_rewards = self.reward(feats)
            rewards = rearrange(flat_rewards, "n 1 -> n")
        else:
            # feats: [B, T, D] -> rewards: [B, T]
            batch_size, traj_len, _ = feats.shape
            flat_feats = rearrange(feats, "b t d -> (b t) d")
            flat_rewards = self.reward(flat_feats)
            rewards = rearrange(flat_rewards, "(b t) 1 -> b t", b=batch_size, t=traj_len)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, device=rewards.device)
            rewards = rewards * mask.to(dtype=rewards.dtype, device=rewards.device)

        return rewards

    @staticmethod
    def init_weights(sequential: nn.Sequential, scales: list[float]) -> None:
        """
        Optional helper (not automatically used).
        Applies orthogonal init to linear layers.
        """
        linear_layers = [m for m in sequential if isinstance(m, nn.Linear)]
        if len(scales) != len(linear_layers):
            raise ValueError(
                f"`scales` length ({len(scales)}) must match number of linear layers ({len(linear_layers)})."
            )
        for layer, gain in zip(linear_layers, scales):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


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
