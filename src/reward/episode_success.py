"""Read task success from pre-reset features of complete attempted episodes."""

from __future__ import annotations

import torch

from reward.terminal_success import TERMINAL_SUCCESS_TERM


def episode_success(
    features: torch.Tensor,
    names: tuple[str, ...] | None,
    mask: torch.Tensor,
    *,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> torch.Tensor | None:
    """Return success [B], or None when the task exposes no success feature.

    Features [B,T,D] and boolean masks/ending flags [B,T] come from complete
    episodes. Terminal success is read only on the last valid transition;
    genuine failure wins even if timeout also fired. Older continuing tasks
    keep their existing ever-reached ``success_bonus`` semantics.
    """
    if features.ndim != 3 or features.shape[:2] != mask.shape:
        raise ValueError("Expected features [B,T,D] and matching episode mask [B,T].")
    for name, value in (
        ("mask", mask),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if value.shape != mask.shape or value.dtype != torch.bool:
            raise ValueError(f"Expected boolean {name} with shape {tuple(mask.shape)}.")
        if value.device != features.device:
            raise ValueError(f"{name} and features must share a device.")
    if names is None:
        return None
    if len(names) != features.shape[-1]:
        raise ValueError("Feature names must match the feature dimension.")
    if TERMINAL_SUCCESS_TERM in names:
        lengths = mask.sum(dim=1)
        if bool((lengths == 0).any()):
            raise ValueError("Terminal success requires nonempty completed episodes.")
        episode_idx = torch.arange(mask.shape[0], device=mask.device)
        last = lengths - 1
        # Paired episode/final-step selection -> [B].
        values = features[episode_idx, last, names.index(TERMINAL_SUCCESS_TERM)]
        return (
            (values > 0.0)
            & truncated[episode_idx, last]
            & ~terminated[episode_idx, last]
        )
    if "success_bonus" in names:
        return ((features[:, :, names.index("success_bonus")] > 0.5) & mask).any(dim=1)
    return None
