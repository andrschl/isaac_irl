"""Projection and proximal operators for reward model regularization.

References:
- L1 ball: Duchi et al., "Efficient Projections onto the L1-Ball for Learning in High Dimensions" (ICML 2008)
- L1 proximal: Parikh & Boyd, "Proximal Algorithms"
- Elastic net proximal: Zou & Hastie, "Regularization and Variable Selection Via the Elastic Net"
"""

from __future__ import annotations

import torch


def project_onto_l2_ball(x: torch.Tensor, radius: float) -> torch.Tensor:
    """Project x onto L2 ball of given radius (in-place style; returns projected tensor).

    min ||x - u||_2  s.t.  ||u||_2 <= radius

    Args:
        x: Tensor of shape [..., D]
        radius: Ball radius (> 0)

    Returns:
        Projected tensor, same shape as x.
    """
    if radius <= 0:
        raise ValueError(f"`radius` must be > 0, got {radius}.")
    norm = x.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
    scale = (norm / radius).clamp(min=1.0)
    return x / scale
