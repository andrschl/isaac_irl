"""Load one experiment: YAML on disk, then command-line overrides."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from config.cli import merge_overrides, parse_overrides
from config.loader import build_dataclass
from config.schema import DirectQCfg, PmdCfg, PpoAlgoCfg, SuccessorQCfg, TrainCfg

_ALGORITHMS = {"ppo": PpoAlgoCfg, "pmd": PmdCfg}
_VALUE_BACKENDS = {"direct": DirectQCfg, "successor": SuccessorQCfg}


def task_slug(task_name: str) -> str:
    """Config-directory name for an Isaac task id."""
    slug = str(task_name).lower()
    for token in ("isaac-", "-v0", "-v1", "-v2"):
        slug = slug.replace(token, "")
    return slug.replace("-", "_")


def load_experiment_cfg(
    *, args: argparse.Namespace, config_root: Path
) -> tuple[TrainCfg, str]:
    """Build a validated :class:`TrainCfg`, and return it with its task name.

    The config file is chosen by ``--config``, else by the task slug under
    ``config_root``, so ``--task Isaac-Lift-Cube-Franka-v0`` reads
    ``configs/lift_cube_franka/experiment.yaml``.
    """
    payload = _read_payload(args=args, config_root=config_root)
    payload = merge_overrides(payload, parse_overrides(list(getattr(args, "set", []))))
    payload = _apply_named_overrides(payload, args)

    algorithm_section = dict(payload.pop("algo", None) or {})
    payload["algorithm"] = _build_algorithm(algorithm_section)

    cfg = build_dataclass(TrainCfg, payload)
    task_name = str(args.task or cfg.env.name or "")
    if not task_name:
        raise ValueError(
            "Task name is required: pass --task or set `env.name` in the config."
        )
    cfg = replace(cfg, env=replace(cfg.env, name=task_name))
    if cfg.reward.learns_a_reward and not cfg.expert_data.path:
        raise ValueError(
            "Learning a reward needs demonstrations: set `expert_data.path` in the "
            "config or pass --expert_data_path."
        )
    return cfg, task_name


def _read_payload(*, args: argparse.Namespace, config_root: Path) -> dict[str, Any]:
    config_path = (
        Path(args.config)
        if getattr(args, "config", None)
        else config_root / task_slug(args.task or "") / "experiment.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Missing experiment config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML at {config_path} must contain a top-level mapping.")
    return payload


def _build_algorithm(section: dict[str, Any]) -> PpoAlgoCfg | PmdCfg:
    """Construct the selected policy algorithm and its value backend.

    Settings that belong to one algorithm live in a block named after it, so a
    config can carry a task's tuning for every algorithm at once and
    ``--algorithm`` selects between them rather than discarding them:

        algo:
          type: pmd
          ppo: {learning_rate: 1.0e-3, gamma: 0.99}
          pmd: {policy_improvements_per_batch: 1, train: {...}}

    Switching algorithms must not quietly fall back to defaults: a benchmark
    where one arm is tuned and the other is not measures the tuning.
    """
    algorithm_type = str(section.pop("type", "ppo"))
    if algorithm_type not in _ALGORITHMS:
        raise ValueError(
            f"`algo.type` must be one of {sorted(_ALGORITHMS)}, got {algorithm_type!r}."
        )
    per_algorithm = {name: dict(section.pop(name, None) or {}) for name in _ALGORITHMS}
    section = merge_overrides(section, per_algorithm[algorithm_type])

    if algorithm_type == "ppo":
        return build_dataclass(PpoAlgoCfg, section, path="algo")
    value = _build_value(dict(section.pop("value", None) or {}))
    algorithm = build_dataclass(PmdCfg, section, path="algo")
    return replace(algorithm, value=value)


def _build_value(section: dict[str, Any]) -> DirectQCfg | SuccessorQCfg:
    """Build the selected value backend from shared plus per-backend settings.

    Keys a backend does not have are an error, so a tuned `latent_dim` can
    never be silently ignored. To let one config hold both tunings - and
    `--value_backend` switch between them - backend-specific settings go in a
    block named after the backend:

        value:
          type: successor
          num_heads: 3          # shared by both
          successor: {latent_dim: 128}
          direct: {target_subset_size: 2}
    """
    value_type = str(section.pop("type", "successor"))
    if value_type not in _VALUE_BACKENDS:
        raise ValueError(
            f"`algo.value.type` must be one of {sorted(_VALUE_BACKENDS)}, "
            f"got {value_type!r}."
        )
    per_backend = {
        name: dict(section.pop(name, None) or {}) for name in _VALUE_BACKENDS
    }
    section.update(per_backend[value_type])
    return build_dataclass(_VALUE_BACKENDS[value_type], section, path="algo.value")


def _apply_named_overrides(
    payload: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Fold the named CLI flags into the YAML payload before it is built."""
    named: dict[str, tuple[str, ...]] = {
        "seed": ("seed",),
        "max_iterations": ("max_iterations",),
        "experiment_name": ("experiment_name",),
        "run_name": ("run_name",),
        "logger": ("logger",),
        "log_project_name": ("wandb_project",),
        "resume": ("resume",),
        "load_run": ("load_run",),
        "checkpoint": ("load_checkpoint",),
        "init_policy_checkpoint": ("init_policy_checkpoint",),
        "num_envs": ("env", "num_envs"),
        "expert_data_path": ("expert_data", "path"),
        "expert_num_trajectories": ("expert_data", "num_trajectories"),
        "reward_source": ("reward", "source"),
        "fit_diagnostic": ("reward", "fit_diagnostic"),
        "bc_weight": ("bc", "weight"),
        "validation_interval": ("runner", "validation_interval"),
        "transition_budget": ("runner", "transition_budget"),
        "algorithm": ("algo", "type"),
    }
    overrides: dict[str, Any] = {}
    for argument_name, path in named.items():
        value = getattr(args, argument_name, None)
        if value is None:
            continue
        node = overrides
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value

    device = getattr(args, "device", None)
    if device is not None:
        overrides["device"] = device
        overrides.setdefault("env", {})["device"] = device

    algorithm_type = str(
        overrides.get("algo", {}).get("type")
        or payload.get("algo", {}).get("type", "ppo")
    )
    # Algorithm-specific YAML wins over shared defaults. Put named CLI flags
    # in that same block so they cannot be overwritten during construction.
    value_backend = getattr(args, "value_backend", None)
    if value_backend is not None:
        if algorithm_type != "pmd":
            raise ValueError("--value_backend requires --algorithm pmd.")
        algorithm = overrides.setdefault("algo", {}).setdefault(algorithm_type, {})
        algorithm.setdefault("value", {})["type"] = value_backend

    learning_rate = getattr(args, "learning_rate", None)
    if learning_rate is not None:
        key = "learning_rate" if algorithm_type == "ppo" else "actor_learning_rate"
        overrides.setdefault("algo", {}).setdefault(algorithm_type, {})[key] = (
            learning_rate
        )

    return merge_overrides(payload, overrides)
