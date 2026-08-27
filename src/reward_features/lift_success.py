"""Lift-task success utilities.

Three concerns share this module because they all wrap the same Isaac Lab
predicate (``object_reached_goal``):

  - :func:`success_bonus_reward` — exposes the predicate as a *reward
    function* (float ``[N]``), so it can be injected as a manager-based
    reward term.
  - :func:`add_success_bonus_term` — mutates an Isaac Lab env config to add
    a ``success_bonus`` reward term backed by :func:`success_bonus_reward`.
  - :func:`build_success_fn` — builds a ``success_fn(env) -> [N] bool``
    closure for the validation success-rate metric, with a probe call to
    confirm the predicate is wired in.
"""

from __future__ import annotations

from typing import Any, Callable

import torch


def _resolve_object_reached_goal() -> Callable[..., torch.Tensor] | None:
    """Import the Isaac predicate only after SimulationApp has started."""
    try:
        from isaaclab_tasks.manager_based.manipulation.lift.mdp.terminations import (
            object_reached_goal,
        )
    except Exception:  # pragma: no cover - Isaac is optional in unit tests
        return None
    return object_reached_goal


def success_bonus_reward(
    env: Any,
    threshold: float = 0.08,
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Return float tensor ``[N]`` of 1.0 where the Lift task's success
    condition is met (object within ``threshold`` of the commanded goal pose),
    else 0.0.

    No-op (zeros) for tasks that don't expose ``object_reached_goal``, so the
    term is safe to inject for non-Lift tasks too — it just becomes a constant
    zero feature.
    """
    object_reached_goal = _resolve_object_reached_goal()
    if object_reached_goal is None:
        return torch.zeros(int(env.num_envs), device=env.device, dtype=torch.float32)
    return object_reached_goal(env, command_name=command_name, threshold=threshold).to(
        torch.float32
    )


def add_success_bonus_term(
    env_cfg: Any,
    *,
    threshold: float,
    weight: float = 0.0,
    command_name: str = "object_pose",
    term_name: str = "success_bonus",
) -> bool:
    """Mutate ``env_cfg.rewards`` to add a success-bonus reward term.

    Must be called BEFORE :func:`gym.make`, since Isaac Lab's reward manager
    introspects ``env_cfg.rewards`` once at env construction.

    ``weight`` defaults to 0.0 — the term is included **only as an IRL
    feature**, not as a contribution to the env reward signal. Combine with
    :attr:`ManagerBasedFeatureCfg.force_include_terms` (add ``term_name`` to
    that set) so the feature extractor doesn't skip the weight-0 term. Set
    ``weight > 0`` if you also want the bonus to enter env rewards (e.g. for
    upstream PPO training).

    Returns ``True`` if the term was added; ``False`` when the env config
    doesn't expose a ``rewards`` section (non-manager-based task) or the
    Isaac Lab ``RewardTermCfg`` class can't be imported.
    """
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        return False
    try:
        from isaaclab.managers import RewardTermCfg as RewTerm
    except Exception:
        return False
    term = RewTerm(
        func=success_bonus_reward,
        params={"threshold": float(threshold), "command_name": command_name},
        weight=float(weight),
    )
    setattr(rewards_cfg, term_name, term)
    return True


def build_success_fn(
    *,
    task_name: str,
    threshold: float,
    probe_env: Any,
) -> Callable[[Any], torch.Tensor] | None:
    """Build a ``success_fn(env) -> [N] bool`` for the validation success metric.

    Returns ``None`` — disabling success logging — for any task that doesn't
    expose ``object_reached_goal``, or if a probe call fails. Best-effort:
    success logging must never break training.
    """
    object_reached_goal = _resolve_object_reached_goal()
    if object_reached_goal is None:
        print(
            f"[WARN] No success metric for task {task_name!r} (object_reached_goal unavailable)."
        )
        return None

    def success_fn(env: Any) -> torch.Tensor:
        return object_reached_goal(env.unwrapped, threshold=threshold)

    try:
        probe = success_fn(probe_env)
        if not isinstance(probe, torch.Tensor):
            raise TypeError(
                f"object_reached_goal returned {type(probe)}, expected Tensor."
            )
    except Exception as exc:
        print(
            f"[WARN] Disabling success metric for task {task_name!r} (probe failed): {exc}"
        )
        return None

    print(f"[INFO] Success metric enabled: object within {threshold} m of goal pose.")
    return success_fn
