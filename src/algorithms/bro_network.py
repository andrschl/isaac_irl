"""Wide residual value network with layer normalization.

The architecture follows the BRO recipe for continuous control: one input
projection, a stack of residual blocks, and a linear head, with a LayerNorm
after every linear layer. The normalization is per-sample, so nothing here
depends on dataset statistics, and it is what keeps a value network from
extrapolating without bound on states and actions it has never seen - the
failure that ends an off-policy run once the policy leaves its own data.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    """``x + f(x)`` for ``[B,W] -> [B,W]`` with normalized, rectified inner layers."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class BroNet(nn.Module):
    """Map ``[B,I]`` to ``[B,O]`` through a normalized residual tower.

    Shapes:
        input: [B, input_dim]
        hidden: [B, width]
        output: [B, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        width: int,
        residual_blocks: int,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, width) <= 0:
            raise ValueError("BroNet dimensions must be positive.")
        if residual_blocks <= 0:
            raise ValueError("residual_blocks must be > 0.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width), nn.LayerNorm(width), nn.ReLU()
        )
        self.blocks = nn.Sequential(
            *(_ResidualBlock(width) for _ in range(residual_blocks))
        )
        self.head = nn.Linear(width, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input [B,{self.input_dim}], got {tuple(values.shape)}."
            )
        return self.head(self.blocks(self.stem(values)))


def reset_module_parameters(module: nn.Module) -> None:
    """Reinitialize every parameter of ``module`` in place.

    Used by periodic resets, which throw away what the networks have learned
    while keeping the replay they learned it from.
    """
    for child in module.modules():
        if child is not module and hasattr(child, "reset_parameters"):
            child.reset_parameters()
