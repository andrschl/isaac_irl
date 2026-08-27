"""Pure-Python CLI override mapping for the Cartpole benchmark."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from config.cartpole_benchmark import (
    CartpoleBenchmarkCfg,
    cartpole_pmd_cfg,
    cartpole_sac_cfg,
)
from config.off_policy import DirectQCfg, SuccessorQCfg


def _override(args: Any, name: str) -> Any:
    """Read one optional override; absent attributes mean "keep the default"."""
    return getattr(args, name, None)


def _value_cfg(name: str | None, *, num_heads: int | None):
    if name == "direct":
        return DirectQCfg(num_heads=num_heads or 5)
    if name == "successor":
        return SuccessorQCfg(num_heads=num_heads or 5)
    return None


def build_cartpole_benchmark_cfg(args: Any) -> CartpoleBenchmarkCfg:
    """Apply a small orthogonal override set without discarding nested defaults."""
    sac = cartpole_sac_cfg()
    pmd = cartpole_pmd_cfg()
    value = _value_cfg(
        _override(args, "value_backend"), num_heads=_override(args, "num_heads")
    )
    if value is not None:
        sac = replace(sac, value=value)
        pmd = replace(pmd, value=value)

    train_overrides = {
        name: value
        for name, value in {
            "updates_per_transition": _override(args, "updates_per_transition"),
            "initial_critic_updates": _override(args, "initial_critic_updates"),
            "critic_learning_rate": _override(args, "critic_learning_rate"),
            "rollout_steps": _override(args, "rollout_steps"),
        }.items()
        if value is not None
    }
    if train_overrides:
        sac = replace(sac, train=replace(sac.train, **train_overrides))
        pmd = replace(pmd, train=replace(pmd.train, **train_overrides))
    actor_learning_rate = _override(args, "actor_learning_rate")
    if actor_learning_rate is not None:
        sac = replace(sac, actor_learning_rate=actor_learning_rate)
        pmd = replace(pmd, actor_learning_rate=actor_learning_rate)
    entropy_temperature = _override(args, "entropy_temperature")
    if entropy_temperature is not None:
        sac = replace(
            sac,
            initial_entropy_temperature=entropy_temperature,
            automatic_entropy_tuning=False,
        )
        pmd = replace(pmd, entropy_temperature=entropy_temperature)
    pmd_overrides = {
        name: value
        for name, value in {
            "beta": _override(args, "pmd_beta"),
            "candidate_target_kl": _override(args, "pmd_candidate_target_kl"),
            "actor_projection_kl": _override(args, "pmd_actor_projection_kl"),
            "projection_steps": _override(args, "pmd_projection_steps"),
            "critic_updates_per_projection": _override(
                args, "pmd_critic_updates_per_projection"
            ),
            "candidate_actions": _override(args, "pmd_candidate_actions"),
            "actor_batch_size": _override(args, "pmd_actor_batch_size"),
        }.items()
        if value is not None
    }
    if pmd_overrides:
        pmd = replace(pmd, **pmd_overrides)

    return CartpoleBenchmarkCfg(
        algorithm=args.algorithm,
        seed=args.seed,
        num_envs=args.num_envs,
        transition_budget=args.transition_budget,
        evaluation_interval_transitions=args.evaluation_interval_transitions,
        evaluation_steps=args.evaluation_steps,
        device=args.device,
        project=args.project,
        group=args.group,
        run_name=args.run_name,
        log_root=args.log_root,
        record_video=args.video,
        video_length=args.video_length,
        action_clip=(
            1.0 if _override(args, "action_clip") is None else args.action_clip
        ),
        sac=sac,
        pmd=pmd,
    )
