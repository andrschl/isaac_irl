"""Finite-horizon terminal success: a goal held to the end of the episode.

Deliberately independent of Isaac. The rule this implements - "the reward is
one on the last transition, and only if the goal was held for the final
``hold_steps`` steps without failing" - is the whole task definition for the
sparse arms, so it is worth being able to exercise it without a simulator.
"""

from __future__ import annotations

import torch

TERMINAL_SUCCESS_TERM = "terminal_success"
"""Reward-term name of the sparse coordinate; learnable only when explicitly selected."""


class TerminalSuccessState:
    """Track a capped goal streak and terminal success for ``N`` environments.

    Shapes: ``streak [N]``, ``success [N]``, ``last_step [N]``, all int64/bool
    on one device. ``success`` is a one-step pulse: it is true only while the
    environment sits on the final transition of a successful episode, and
    :meth:`reset` clears it when Isaac recycles that lane.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        horizon: int,
        hold_steps: int,
        device: str | torch.device,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"Expected num_envs > 0, got {num_envs}.")
        if not 0 < hold_steps <= horizon:
            raise ValueError(
                f"Expected 0 < hold_steps <= horizon, got {hold_steps}, {horizon}."
            )
        self.horizon = int(horizon)
        self.hold_steps = int(hold_steps)
        self.streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Which physical step each lane last consumed, so a second evaluation
        # within one step cannot advance the streak twice.
        self.last_step = torch.full_like(self.streak, -1)

    def update(
        self,
        reached: torch.Tensor,
        failed: torch.Tensor,
        ages: torch.Tensor,
        *,
        step: int,
    ) -> None:
        """Advance the streak by one physical step.

        Args:
            reached: [N] bool, the goal condition holds after this step.
            failed: [N] bool, a genuine termination fired; failure beats success.
            ages: [N] episode length after this step, 1-based.
            step: monotonic physical step counter, used only for idempotence.
        """
        self._require_matching(reached=reached, failed=failed, ages=ages)
        if reached.dtype != torch.bool or failed.dtype != torch.bool:
            raise ValueError("reached and failed must be boolean [N] tensors.")
        # A lane at age zero has not taken a step in its current episode, and a
        # lane already advanced at this step must not be advanced again.
        active = (ages > 0) & (self.last_step != step)
        next_streak = torch.where(
            reached, (self.streak + 1).clamp_max(self.hold_steps), 0
        )
        self.streak[active] = next_streak[active]
        final = (ages == self.horizon) & (self.streak == self.hold_steps) & ~failed
        self.success[active] = final[active]
        self.last_step[active] = step

    def _require_matching(self, **tensors: torch.Tensor) -> None:
        for name, value in tensors.items():
            if value.shape != self.streak.shape or value.device != self.streak.device:
                raise ValueError(
                    f"Expected {name} {tuple(self.streak.shape)} on "
                    f"{self.streak.device}, got {tuple(value.shape)} on {value.device}."
                )

    def reset(self, env_ids: torch.Tensor | slice) -> None:
        """Clear the streak and the success pulse for the given lanes."""
        self.streak[env_ids] = 0
        self.success[env_ids] = False
        self.last_step[env_ids] = -1

    def observations(self, ages: torch.Tensor) -> torch.Tensor:
        """Remaining time and capped hold progress. Shape: [N, 2].

        Read-only: a finite-horizon value function is non-stationary, so the
        student has to be told where in the episode it is, but reading that off
        must not advance the task state - the observation manager computes
        every term twice per step (once pre-reset for the true terminal
        observation, once after).
        """
        self._require_matching(ages=ages)
        remaining = (1.0 - ages.to(torch.float32) / self.horizon).clamp(0.0, 1.0)
        progress = self.streak.to(torch.float32) / float(self.hold_steps)
        return torch.stack((remaining, progress), dim=-1)
