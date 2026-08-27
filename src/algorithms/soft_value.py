"""Bellman targets shared by direct soft Q and entropy-return critics."""

from __future__ import annotations

import torch


def entropy_return_td_target(
    next_log_probability: torch.Tensor,
    target_entropy_q: torch.Tensor,
    bootstrap_valid: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Return ``gamma * m * (-log pi(a'|s') + Q_H^-)`` with shape ``[B]``."""
    expected_shape = next_log_probability.shape
    if next_log_probability.ndim != 1 or target_entropy_q.shape != expected_shape:
        raise ValueError("Entropy target inputs must have matching [B] shapes.")
    if bootstrap_valid.shape != expected_shape:
        raise ValueError("bootstrap_valid must match entropy target inputs [B].")
    return (
        gamma
        * bootstrap_valid.to(next_log_probability.dtype)
        * (-next_log_probability + target_entropy_q)
    )


def soft_task_q_td_target(
    rewards: torch.Tensor,
    next_log_probability: torch.Tensor,
    target_soft_q: torch.Tensor,
    bootstrap_valid: torch.Tensor,
    *,
    gamma: float,
    entropy_coefficient: float,
) -> torch.Tensor:
    """Return the standard soft task-Q Bellman target with shape ``[B]``."""
    if rewards.ndim != 1:
        raise ValueError(f"Expected rewards [B], got {tuple(rewards.shape)}.")
    if (
        next_log_probability.shape != rewards.shape
        or target_soft_q.shape != rewards.shape
        or bootstrap_valid.shape != rewards.shape
    ):
        raise ValueError("Soft task-Q target inputs must have matching [B] shapes.")
    if entropy_coefficient < 0.0:
        raise ValueError("entropy_coefficient must be >= 0.")
    continuation = target_soft_q - entropy_coefficient * next_log_probability
    return rewards + gamma * bootstrap_valid.to(rewards.dtype) * continuation
