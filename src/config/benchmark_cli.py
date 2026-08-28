"""Pure-Python CLI override mapping for the fixed-reward benchmark."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from config.benchmark import BenchmarkTask, FixedRewardBenchmarkCfg, build_task
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


def _apply_task_overrides(task: BenchmarkTask, args: Any) -> BenchmarkTask:
    overrides = {
        name: value
        for name, value in {
            "num_envs": _override(args, "num_envs"),
            "transition_budget": _override(args, "transition_budget"),
            "evaluation_interval_transitions": _override(
                args, "evaluation_interval_transitions"
            ),
            "evaluation_steps": _override(args, "evaluation_steps"),
            "policy_action_bound": _override(args, "policy_action_bound"),
        }.items()
        if value is not None
    }
    return replace(task, **overrides) if overrides else task


def _apply_algorithm_overrides(task: BenchmarkTask, args: Any) -> BenchmarkTask:
    sac, pmd = task.sac, task.pmd
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
            "gamma": _override(args, "gamma"),
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

    gamma = _override(args, "gamma")
    ppo = replace(task.ppo, gamma=gamma) if gamma is not None else task.ppo
    return replace(task, sac=sac, pmd=pmd, ppo=ppo)


def build_benchmark_cfg(args: Any) -> FixedRewardBenchmarkCfg:
    """Resolve a task profile and apply a small orthogonal override set."""
    task = build_task(getattr(args, "task", "cartpole"))
    task = _apply_algorithm_overrides(_apply_task_overrides(task, args), args)
    return FixedRewardBenchmarkCfg(
        algorithm=args.algorithm,
        task=task,
        seed=_override(args, "seed") or 1,
        device=_override(args, "device") or "cuda:0",
        project=_override(args, "project") or "isaac-irl-sample-efficiency",
        group=_override(args, "group") or f"{task.name}-fixed-reward-v4",
        run_name=_override(args, "run_name"),
        log_root=_override(args, "log_root") or "logs/benchmark",
        record_video=bool(_override(args, "video")),
        video_length=_override(args, "video_length") or 250,
    )
