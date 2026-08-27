from __future__ import annotations

import torch
import torch.nn as nn


class BaseRewardModel(nn.Module):
    """Base reward API used by the training pipeline."""

    def reset(self, dones=None):
        del dones

    def get_reward_from_features(
        self,
        feats: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-step reward from features.

        Args:
            feats:
                Features with shape [N, D] or [B, T, D].
            mask:
                Optional mask with shape [N] or [B, T].
        """
        del feats, mask
        raise NotImplementedError

    def discounted_returns_from_features(
        self,
        feats: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Evaluate discounted reward returns.

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
        rewards = self.get_reward_from_features(feats, mask)  # [B, T]
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, device=model_device, dtype=model_dtype)
        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if rewards.ndim != 2:
            raise ValueError(
                "Expected reward model to return rewards with shape [B, T], "
                f"got {tuple(rewards.shape)}."
            )
        if rewards.shape != mask.shape:
            raise ValueError(
                "Expected reward model output shape to match mask shape [B, T], "
                f"got rewards {tuple(rewards.shape)} and mask {tuple(mask.shape)}."
            )
        time_steps = rewards.shape[1]
        powers = gamma ** torch.arange(
            time_steps, device=rewards.device, dtype=rewards.dtype
        )
        return torch.einsum("bt,bt,t->b", rewards, mask.to(dtype=rewards.dtype), powers)

    def get_regularization_loss(self) -> torch.Tensor:
        """L2 regularization penalty (for differentiable reg). No-op by default."""
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return torch.tensor(0.0, device=device)

    def apply_proximal_step(self) -> None:
        """Apply L1/elastic net proximal operator to parameters. No-op by default."""
        pass

    def project_weights(self) -> None:
        """Project linear weights onto norm ball (linear models only). No-op by default."""
        pass

    @staticmethod
    def _validate_discount_inputs(feats: torch.Tensor, mask: torch.Tensor) -> None:
        if feats.ndim != 3:
            raise ValueError(
                f"Expected feats shape [B, T, D], got {tuple(feats.shape)}."
            )
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape [B, T], got {tuple(mask.shape)}.")
        if feats.shape[:2] != mask.shape:
            raise ValueError(
                "Expected feats leading shape [B, T] to match mask shape [B, T], "
                f"got feats {tuple(feats.shape)} and mask {tuple(mask.shape)}."
            )

    def _parameter_device_dtype(
        self, *, fallback: torch.Tensor
    ) -> tuple[torch.device, torch.dtype]:
        param = next(self.parameters(), None)
        if param is None:
            return fallback.device, fallback.dtype
        return param.device, param.dtype
