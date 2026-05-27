from __future__ import annotations

from typing import Any

import torch

from algorithms.bc import BC
from interfaces import EnvTransition


class _RslRlActor:
    """Adapt an rsl_rl MLPModel to the core ``Actor`` protocol.

    `act` samples stochastically via the model's distribution head;
    `act_inference` returns the deterministic mean.
    """

    def __init__(self, mlp_actor: Any) -> None:
        self._actor = mlp_actor

    def act(self, obs: Any) -> torch.Tensor:
        return self._actor(obs, stochastic_output=True)

    def act_inference(self, obs: Any) -> torch.Tensor:
        return self._actor(obs)


class RslRlPpoAdapter:
    """Adapt rsl_rl PPO (and PPOWithBC) to the runner ``RlAlgorithm`` API."""

    def __init__(self, rl_alg: Any, *, bc_alg: BC | None = None) -> None:
        self._rl_alg = rl_alg
        self._bc_alg = bc_alg
        self._last_bc_val_loss = float("nan")
        mlp_actor = getattr(rl_alg, "actor", None)
        if mlp_actor is None:
            raise AttributeError(
                "RslRlPpoAdapter requires `rl_alg.actor` (rsl_rl >= 4.0 MLPModel contract)."
            )
        self._actor_facade = _RslRlActor(mlp_actor)

    @property
    def _alpha(self) -> float:
        return 0.0 if self._bc_alg is None else float(self._bc_alg.cfg.alpha)

    @property
    def _is_bc_only(self) -> bool:
        return self._bc_alg is not None and self._alpha >= 1.0

    @property
    def collects_rollouts(self) -> bool:
        return not self._is_bc_only

    @property
    def actor(self) -> _RslRlActor:
        """Side-effect-free policy facade for validation / imitator rollouts."""
        return self._actor_facade

    def collect_action(self, obs: Any) -> torch.Tensor:
        return self._rl_alg.act(obs)

    def observe(self, transition: EnvTransition) -> None:
        self._rl_alg.process_env_step(
            transition.next_obs,
            transition.rewards,
            transition.dones,
            transition.extras,
        )

    def end_rollout(self, last_obs: Any) -> None:
        if self._is_bc_only:
            return
        self._rl_alg.compute_returns(last_obs)

    def update(self) -> dict[str, float]:
        result = self._rl_alg.bc_only_update() if self._is_bc_only else self._rl_alg.update()

        metrics: dict[str, float] = {}
        if isinstance(result, dict):
            if "surrogate" in result:
                metrics["RL/policy_loss"] = float(result["surrogate"])
            if "bc" in result:
                metrics["BC/loss"] = float(result["bc"])
        else:
            metrics["RL/policy_loss"] = float("nan")
        return metrics

    def train_metrics(self) -> dict[str, float]:
        metrics = self._policy_std_stats()
        if self._bc_alg is not None:
            metrics["BC/alpha"] = self._alpha
        return metrics

    def eval_metrics(self) -> dict[str, float]:
        if self._bc_alg is None:
            return {}
        actor = getattr(self._rl_alg, "actor", None)
        if actor is None:
            return {}
        val_loss = self._bc_alg.compute_validation_loss(actor)
        if val_loss is None:
            return {}
        self._last_bc_val_loss = float(val_loss)
        return {"BC/val_loss": self._last_bc_val_loss}

    def train_mode(self) -> None:
        self._rl_alg.train_mode()

    def eval_mode(self) -> None:
        self._rl_alg.eval_mode()

    def save_state(self) -> dict[str, Any]:
        return dict(self._rl_alg.save())

    def load_state(self, checkpoint: dict[str, Any], *, load_optimizer: bool = True) -> None:
        load_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": bool(load_optimizer),
            "iteration": True,
            "rnd": True,
        }
        self._rl_alg.load(checkpoint, load_cfg=load_cfg, strict=True)

    def _policy_std_stats(self) -> dict[str, float]:
        actor = getattr(self._rl_alg, "actor", None)
        dist = getattr(actor, "distribution", None) if actor is not None else None
        if dist is None:
            return {}
        with torch.no_grad():
            if hasattr(dist, "std_param"):
                std = dist.std_param.detach()
            elif hasattr(dist, "log_std_param"):
                std = dist.log_std_param.detach().exp()
            else:
                return {}
        return {
            "RL/policy_std_min": float(std.min().item()),
            "RL/policy_std_max": float(std.max().item()),
            "RL/policy_std_mean": float(std.mean().item()),
        }
