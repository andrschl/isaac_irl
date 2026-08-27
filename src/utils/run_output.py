"""Serialize and write per-run output to disk.

Used by the training script to dump the resolved ``TrainCfg`` (and the
ground-truth env reward weights, when available) as YAML under
``<log_dir>/params/``. Pure side-effect-free helpers for nested dataclasses
plus the I/O entry point :func:`dump_run_configs`.
"""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            str(field_def.name): _to_serializable(getattr(value, field_def.name))
            for field_def in fields(value)
        }

    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(item) for item in value]

    if isinstance(value, torch.dtype):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _dump_yaml(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handle:
        yaml.safe_dump(_to_serializable(payload), file_handle, sort_keys=False)


def _is_scalar_like(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _try_get_nested_attr(root: Any, attr_path: tuple[str, ...]) -> Any | None:
    current = root
    for attr_name in attr_path:
        if not hasattr(current, attr_name):
            return None
        current = getattr(current, attr_name)
    if _is_scalar_like(current):
        return current
    return None


def extract_ground_truth_reward_weights(env: Any) -> dict[str, float]:
    """Extract reward term weights from env's reward_manager (ground truth params)."""
    try:
        unwrapped = getattr(env, "unwrapped", env)
        reward_manager = getattr(unwrapped, "reward_manager", None)
        if reward_manager is None:
            return {}
        term_names = getattr(reward_manager, "_term_names", [])
        term_cfgs = getattr(reward_manager, "_term_cfgs", [])
        if len(term_names) != len(term_cfgs):
            return {}
        return {
            str(name): float(getattr(cfg, "weight", 0.0))
            for name, cfg in zip(term_names, term_cfgs)
        }
    except Exception:
        return {}


def _summarize_env_cfg(env_cfg: Any, train_cfg: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "cfg_type": f"{type(env_cfg).__module__}.{type(env_cfg).__name__}",
        "task_name": str(train_cfg.env.name),
        "train_device": str(train_cfg.env.device),
        "train_num_envs": train_cfg.env.num_envs,
    }

    for key_name, attr_path in (
        ("sim_device", ("sim", "device")),
        ("scene_num_envs", ("scene", "num_envs")),
        ("seed", ("seed",)),
        ("decimation", ("decimation",)),
        ("episode_length_s", ("episode_length_s",)),
    ):
        value = _try_get_nested_attr(env_cfg, attr_path)
        if value is not None:
            summary[key_name] = value

    return summary


def dump_run_configs(
    log_dir: str,
    env_cfg: Any,
    train_cfg: Any,
    env: Any | None = None,
) -> None:
    """Write ``<log_dir>/params/experiment.yaml`` and optionally
    ``ground_truth_reward_weights.yaml`` (when ``env`` exposes a reward manager).
    """
    serializable_train_cfg = _to_serializable(train_cfg)
    params_dir = os.path.join(log_dir, "params")
    env_summary = _summarize_env_cfg(env_cfg, train_cfg)
    experiment_cfg = {**serializable_train_cfg, "env": env_summary}
    _dump_yaml(os.path.join(params_dir, "experiment.yaml"), experiment_cfg)
    if env is not None:
        ground_truth = extract_ground_truth_reward_weights(env)
        if ground_truth:
            _dump_yaml(
                os.path.join(params_dir, "ground_truth_reward_weights.yaml"),
                ground_truth,
            )
            weights_str = "  ".join(f"{k}={v:.4f}" for k, v in ground_truth.items())
            print(f"[INFO] Ground truth reward weights: {weights_str}")
