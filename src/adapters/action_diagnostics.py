"""Action-range diagnostics shared by fixed-reward algorithm adapters."""

from __future__ import annotations

import torch


def bounded_action_metrics(actions: torch.Tensor, *, bound: float) -> dict[str, float]:
    """Summarize the action distribution of one rollout ``[..., A]``."""
    if actions.ndim < 2:
        raise ValueError(f"Expected actions [...,A], got {tuple(actions.shape)}.")
    if bound <= 0.0:
        raise ValueError(f"bound must be > 0, got {bound}.")
    if not torch.isfinite(actions).all():
        raise ValueError("Action diagnostics received non-finite actions.")

    flat = actions.detach().float().flatten()
    quantiles = torch.quantile(
        flat, torch.tensor([0.01, 0.5, 0.99], device=flat.device)
    )
    tolerance = _tolerance(bound)
    return {
        "Action/q01": float(quantiles[0].item()),
        "Action/median": float(quantiles[1].item()),
        "Action/q99": float(quantiles[2].item()),
        "Action/abs_mean": float(flat.abs().mean().item()),
        "Action/boundary_saturation_fraction": float(
            flat.abs().ge(bound - tolerance).float().mean().item()
        ),
        "Action/bound": float(bound),
    }


def clipping_metrics(
    raw_actions: torch.Tensor,
    executed_actions: torch.Tensor,
    *,
    bound: float,
) -> dict[str, float]:
    """Report how much of an unbounded policy's output the environment clipped.

    Only meaningful where raw and executed actions can genuinely differ; a
    tanh-bounded policy is inside the bound by construction.
    """
    if raw_actions.shape != executed_actions.shape or raw_actions.ndim < 2:
        raise ValueError(
            "Expected matching raw/executed action shapes [...,A], got "
            f"{tuple(raw_actions.shape)} and {tuple(executed_actions.shape)}."
        )
    metrics = bounded_action_metrics(executed_actions, bound=bound)
    metrics["Action/raw_abs_mean"] = float(
        raw_actions.detach().float().abs().mean().item()
    )
    metrics["Action/clipped_fraction"] = float(
        (raw_actions.detach() - executed_actions.detach())
        .abs()
        .gt(_tolerance(bound))
        .float()
        .mean()
        .item()
    )
    return metrics


def _tolerance(bound: float) -> float:
    return max(1.0e-6, bound * 1.0e-3)
