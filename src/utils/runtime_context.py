from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeContext:
    """Shared runtime context for feature-buffer IRL wiring."""

    num_envs: int
    feature_dim: int
    device: str = "cpu"
