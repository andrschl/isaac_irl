from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch.nn as nn

from algorithms.bc import BC


def make_ppo_with_bc_cls(ppo_cls: type) -> type:
    """Return a subclass of ``ppo_cls`` that adds a behavioral-cloning term.

    Why a factory instead of a plain ``class PPOWithBC(rsl_rl...PPO)``:
    importing ``rsl_rl.algorithms.PPO`` at module load time would pull rsl_rl
    in before Isaac Sim's ``SimulationApp`` is created — which is forbidden by
    the Carbonite framework's plugin system. The training script imports
    ``algorithms`` early (for ``IRL`` / ``BC``) and only constructs the
    SimulationApp later, so we defer the PPO import to runtime by accepting
    the class here.

    The subclass overrides :meth:`update` to inject a BC gradient at every
    optimizer step. Just before each ``optimizer.step()`` call that PPO would
    normally do, we:

      1. Scale actor PPO gradients in-place by ``(1 - alpha)``.
      2. Compute a BC loss on a fresh expert minibatch and call
         ``(alpha * bc_loss).backward()`` to accumulate BC actor gradients on top.
      3. Re-clip the combined actor gradient norm to ``max_grad_norm``.
      4. Call the original ``optimizer.step()``.

    The critic keeps the normal PPO critic update. The resulting update on
    the actor parameters is

        Δθ ∝ (1 - α) · ∇L_PPO  +  α · ∇L_BC

    which is exactly the gradient of the convex combination

        L = (1 - α) · L_PPO + α · L_BC

    For pure-BC mode (``alpha == 1``) the RL agent adapter calls
    :meth:`bc_only_update` directly instead of :meth:`update`, since PPO's
    rollout storage is not populated when rollouts are skipped.
    """

    class PPOWithBC(ppo_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, bc_alg: BC, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.bc_alg = bc_alg
            if not isinstance(getattr(self, "actor", None), nn.Module):
                raise AttributeError(
                    "PPOWithBC requires the wrapped PPO to expose `self.actor` "
                    "(rsl_rl >= 4.0 MLPModel contract)."
                )

        @property
        def bc_weight(self) -> float:
            return self.bc_alg.cfg.weight

        @contextlib.contextmanager
        def _inject_bc_into_optimizer_step(self, alpha: float) -> Iterator[list[float]]:
            """Register a pre-step hook on ``self.optimizer`` so that every
            ``optimizer.step()`` call:

              1. scales accumulated actor PPO gradients by ``(1 - alpha)``,
              2. backprops ``alpha * L_BC`` onto the actor on top,
              3. re-clips the combined actor gradient norm to ``max_grad_norm``,

            before the optimizer's own ``step`` executes.

            Note: rsl_rl clips PPO grads *before* calling ``optimizer.step()``,
            so the effective update is ``clip((1-α)·clip(g_ppo) + α·g_bc)``
            rather than ``clip((1-α)·g_ppo + α·g_bc)``.

            Yields a list that receives one ``L_BC`` scalar per step, so the
            caller can average it after the wrapped update returns.
            """
            actor_params = list(self.actor.parameters())
            scale = 1.0 - alpha
            max_grad_norm = self.max_grad_norm
            bc_losses: list[float] = []

            def pre_step_hook(optimizer, args, kwargs):
                del optimizer, args, kwargs
                for param in actor_params:
                    if param.grad is not None:
                        param.grad.mul_(scale)

                bc_loss = self.bc_alg.compute_loss(self.actor)
                (alpha * bc_loss).backward()
                bc_losses.append(bc_loss.detach().item())

                nn.utils.clip_grad_norm_(actor_params, max_grad_norm)

            handle = self.optimizer.register_step_pre_hook(pre_step_hook)
            try:
                yield bc_losses
            finally:
                handle.remove()

        def update(self) -> dict[str, float]:
            if self.bc_weight == 0.0:
                return super().update()

            alpha = self.bc_weight
            with self._inject_bc_into_optimizer_step(alpha) as bc_losses:
                result = super().update()

            if not isinstance(result, dict):
                result = {"surrogate": float("nan")}
            if bc_losses:
                result["bc"] = sum(bc_losses) / len(bc_losses)
            result["bc_weight"] = alpha
            return result

        def bc_only_update(self) -> dict[str, float]:
            """Run a single BC-only gradient step on the actor.

            Used by the runner when ``alpha == 1`` (rollouts disabled). The
            ``alpha`` weight is not applied here — at alpha=1 we want the full
            BC gradient.
            """
            bc_loss = self.bc_alg.compute_loss(self.actor)
            self.optimizer.zero_grad(set_to_none=True)
            bc_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            return {"bc": bc_loss.detach().item(), "bc_weight": self.bc_weight}

    PPOWithBC.__name__ = f"{ppo_cls.__name__}WithBC"
    PPOWithBC.__qualname__ = PPOWithBC.__name__
    return PPOWithBC
