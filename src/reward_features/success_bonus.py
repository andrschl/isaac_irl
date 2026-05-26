"""Custom reward term: constant bonus when the Lift task's success predicate
fires (object within ``threshold`` of the commanded goal pose).

Exposes Isaac Lab's :func:`object_reached_goal` as a *reward function* (float
tensor) rather than a termination, plus a helper that mutates an Isaac Lab
env config in place to add the new term. The injected term becomes both a
reward-manager term (so PPO sees the bonus in env rewards) and an IRL feature
(:mod:`reward_features.manager_based` picks up every non-zero-weight term).
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from isaaclab_tasks.manager_based.manipulation.lift.mdp import object_reached_goal
except Exception:  # pragma: no cover - isaac is optional at import time
    object_reached_goal = None  # type: ignore[assignment]


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
