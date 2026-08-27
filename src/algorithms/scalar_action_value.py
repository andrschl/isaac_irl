"""Scalar action-value models for entropy and fixed-task references."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


def _action_value_mlp(
    observation_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
) -> nn.Sequential:
    if min(observation_dim, action_dim, output_dim) <= 0:
        raise ValueError("Action-value dimensions must be positive.")
    if not hidden_dims or any(hidden_dim <= 0 for hidden_dim in hidden_dims):
        raise ValueError("hidden_dims must contain positive dimensions.")
    dimensions = (observation_dim + action_dim, *hidden_dims, output_dim)
    layers: list[nn.Module] = []
    for input_dim, next_dim in zip(dimensions[:-2], dimensions[1:-1], strict=True):
        layers.extend((nn.Linear(input_dim, next_dim), nn.ELU()))
    layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
    return nn.Sequential(*layers)


class ScalarActionValue(nn.Module):
    """Map observations ``[B,O]`` and actions ``[B,A]`` to values ``[B]``."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.network = _action_value_mlp(observation_dim, action_dim, hidden_dims, 1)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(
                f"Expected observations [B,{self.observation_dim}], "
                f"got {tuple(observations.shape)}."
            )
        if tuple(actions.shape) != (observations.shape[0], self.action_dim):
            raise ValueError(
                f"Expected actions [B,{self.action_dim}], got {tuple(actions.shape)}."
            )
        return self.network(torch.cat((observations, actions), dim=-1)).squeeze(-1)

    def target_copy(self) -> "ScalarActionValue":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        target.eval()
        return target


class ScalarActionValueEnsemble(nn.Module):
    """Independent scalar action-value heads, returning all values ``[M,B]``."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        num_heads: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        self.heads = nn.ModuleList(
            [
                ScalarActionValue(observation_dim, action_dim, hidden_dims=hidden_dims)
                for _ in range(num_heads)
            ]
        )
        self.num_heads = int(num_heads)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([head(observations, actions) for head in self.heads], dim=0)

    def head(
        self, head_index: int, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if not 0 <= head_index < self.num_heads:
            raise ValueError(
                f"head_index must be in [0,{self.num_heads}), got {head_index}."
            )
        return self.heads[head_index](observations, actions)

    def mean(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(observations, actions).mean(dim=0)

    def target_copy(self) -> "ScalarActionValueEnsemble":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        target.eval()
        return target


@torch.no_grad()
def polyak_update_scalar_value(
    source: nn.Module, target: nn.Module, coefficient: float
) -> None:
    if not 0.0 < coefficient <= 1.0:
        raise ValueError("coefficient must be in (0,1].")
    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        target_parameter.lerp_(source_parameter, coefficient)
