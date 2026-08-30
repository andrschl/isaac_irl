"""Checkpoint helpers for rsl_rl actor models."""

from __future__ import annotations

from typing import Any

import torch


def load_actor_only_checkpoint(
    actor: Any, checkpoint_path: str, *, device: str | torch.device
) -> None:
    """Load actor parameters without restoring any other training state.

    Reads either an rsl_rl actor (``actor_state_dict``) or the bounded actor an
    off-policy run saves at ``rl_alg_state.learner.policy`` — the second is what
    makes a run start from a policy this repository trained, which is how a
    behaviour-cloned or previously fine-tuned actor gets into a new run.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor_state = _find_actor_state(checkpoint.get("rl_alg_state", checkpoint))
    if actor_state is None:
        raise ValueError(
            "Actor-only initialization found no actor in the checkpoint: "
            "expected actor_state_dict, actor, or learner.policy under "
            "rl_alg_state."
        )
    actor.load_state_dict(actor_state, strict=True)


def _find_actor_state(algorithm_state: dict[str, Any]) -> dict[str, Any] | None:
    """Locate the actor parameters in either checkpoint layout."""
    for key in ("actor_state_dict", "actor"):
        state = algorithm_state.get(key)
        if state is not None:
            return state
    learner = algorithm_state.get("learner")
    if isinstance(learner, dict):
        return learner.get("policy")
    return None
