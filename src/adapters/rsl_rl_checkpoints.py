"""Checkpoint helpers for rsl_rl actor models."""

from __future__ import annotations

from typing import Any

import torch


def load_actor_only_checkpoint(
    actor: Any, checkpoint_path: str, *, device: str | torch.device
) -> None:
    """Load actor parameters without restoring any other training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    algorithm_state = checkpoint.get("rl_alg_state", checkpoint)
    actor_state = algorithm_state.get("actor_state_dict")
    if actor_state is None:
        actor_state = algorithm_state.get("actor")
    if actor_state is None:
        raise ValueError(
            "Actor-only initialization expected rl_alg_state.actor_state_dict "
            "(or actor_state_dict at the checkpoint root)."
        )
    actor.load_state_dict(actor_state, strict=True)
