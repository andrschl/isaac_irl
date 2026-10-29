"""Complete behavior-policy episodes and their actual interaction cost."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from einops import rearrange

from storage.observations import map_observation
from storage.transition_replay import TransitionRollout


@dataclass(frozen=True, slots=True)
class EpisodeBatch:
    """Transitions [B,T,...] and valid-prefix mask [B,T], one episode per row.

    Padding is not experience. environment_transitions also counts simulated
    lanes stepped after their selected episode ended or outside the quota.
    """

    transitions: TransitionRollout
    mask: torch.Tensor
    environment_transitions: int
    collection_seconds: float = 0.0
    reset_seconds: float = 0.0

    def __post_init__(self) -> None:
        expected = self.transitions.actions.shape[:2]
        if self.mask.shape != expected or self.mask.dtype != torch.bool:
            raise ValueError(
                f"Expected boolean episode mask {expected}, got {self.mask.shape} / {self.mask.dtype}."
            )
        if self.mask.device != self.transitions.actions.device:
            raise ValueError("Episode mask and transitions must share a device.")
        if not torch.equal(
            self.mask, self.mask.to(torch.int64).cummin(dim=1).values.bool()
        ):
            raise ValueError("Episode masks must be contiguous valid prefixes.")
        lengths = self.mask.sum(dim=1)
        if bool((lengths == 0).any()):
            raise ValueError("Every episode must contain at least one transition.")
        done = self.transitions.terminated | self.transitions.truncated
        last = (
            torch.arange(done.shape[1], device=done.device)[None, :]
            == lengths[:, None] - 1
        )
        if not torch.equal(done & self.mask, last):
            raise ValueError(
                "Every episode must end exactly once, on its last valid transition."
            )
        if self.environment_transitions < self.transition_count:
            raise ValueError(
                "Actual environment transitions cannot be fewer than stored transitions."
            )

    @property
    def episode_count(self) -> int:
        return self.mask.shape[0]

    @property
    def transition_count(self) -> int:
        return int(self.mask.sum().item())

    @property
    def returns(self) -> torch.Tensor:
        """Undiscounted environment return per attempted episode [B]."""
        return (self.transitions.environment_rewards * self.mask).sum(dim=1)

    def valid_rollout(self) -> TransitionRollout:
        """Flatten valid transitions to [1,V,...] for TD replay; no padding."""

        def select(tensor):
            return rearrange(tensor[self.mask], "v ... -> 1 v ...")

        values = {}
        for field in fields(TransitionRollout):
            value = getattr(self.transitions, field.name)
            values[field.name] = (
                value
                if field.name in {"policy_version", "rollout_id"}
                else map_observation(value, select)
            )
        return TransitionRollout(**values)
