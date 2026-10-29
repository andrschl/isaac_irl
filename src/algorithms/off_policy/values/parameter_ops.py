"""Target-network construction and interpolation shared by every critic."""

from __future__ import annotations

import copy
from typing import TypeVar

import torch
import torch.nn as nn

ModuleT = TypeVar("ModuleT", bound=nn.Module)


def frozen_target_copy(module: ModuleT) -> ModuleT:
    """Return an evaluation-mode copy of ``module`` that never takes gradients."""
    target = copy.deepcopy(module)
    target.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, coefficient: float) -> None:
    """Move ``target`` parameters ``coefficient`` of the way toward ``source``."""
    if not 0.0 < coefficient <= 1.0:
        raise ValueError(f"coefficient must be in (0,1], got {coefficient}.")
    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        target_parameter.lerp_(source_parameter, coefficient)
