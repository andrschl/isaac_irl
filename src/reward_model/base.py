from __future__ import annotations

import torch
import torch.nn as nn


class BaseRewardModel(nn.Module):
    """Base reward API used by the training pipeline."""

    def reset(self, dones=None):
        del dones

    @property
    def is_linear(self) -> bool:
        """Whether reward evaluation is linear in features."""
        return False

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
