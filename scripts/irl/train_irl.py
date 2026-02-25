# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train IRL agent with PPO + feature-buffer IRL."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from einops import rearrange

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from algorithms import IRL, IRLCfg
from reward_features.manager_based import ManagerBasedFeatureCfg, manager_based_reward_features
from reward_model import RewardModel, RewardModelCfg
from runner import IrlRunner, IrlRunnerCfg
from utils.runtime_context import RuntimeContext
from storage.feature_storage import FeatureBufCfg


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@dataclass(slots=True)
class ActorCriticCfg:
    actor_hidden_dims: tuple[int, ...] = (256, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (256, 128, 64)
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    state_dependent_std: bool = False
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False


@dataclass(slots=True)
class PpoAlgoCfg:
    learning_rate: float = 1.0e-4
    gamma: float = 0.98
    lam: float = 0.95
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    entropy_coef: float = 0.006
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    normalize_advantage_per_mini_batch: bool = False
    rnd_cfg: dict | None = None
    symmetry_cfg: dict | None = None


@dataclass(slots=True)
class EnvCfg:
    name: str | None = None
    device: str = "cuda:0"
    num_envs: int | None = None


@dataclass(slots=True)
class TrainCfg:
    experiment_name: str = "default_experiment"
    run_name: str = ""
    logger: str = "tensorboard"
    wandb_project: str = "isaaclab"
    neptune_project: str = "isaaclab"

    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"

    seed: int = 42
    max_iterations: int = 1500
    device: str = "cuda:0"
    env: EnvCfg = field(default_factory=EnvCfg)

    feature_map: ManagerBasedFeatureCfg = field(default_factory=ManagerBasedFeatureCfg)
    irl: IRLCfg = field(default_factory=IRLCfg)
    runner: IrlRunnerCfg = field(default_factory=IrlRunnerCfg)
    reward: RewardModelCfg = field(default_factory=lambda: RewardModelCfg(num_features=1))
    actor_critic: ActorCriticCfg = field(default_factory=ActorCriticCfg)
    rl_algorithm: PpoAlgoCfg = field(default_factory=PpoAlgoCfg)


@dataclass(slots=True)
class RuntimeDeps:
    AppLauncher: Any
    gym: Any
    parse_env_cfg: Callable[..., Any]
    get_checkpoint_path: Callable[..., str]
    RslRlVecEnvWrapper: Any
    PPO: Any
    ActorCritic: Any
    resolve_obs_groups: Callable[..., Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _task_slug(task_name: str) -> str:
    slug = str(task_name).lower()
    for token in ("isaac-", "-v0", "-v1", "-v2"):
        slug = slug.replace(token, "")
    return slug.replace("-", "_")


def _to_tuple_ints(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    raise TypeError(f"Expected sequence of ints, got {type(value)!r}.")


def _to_mapping(payload: Any, *, section_name: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config section '{section_name}' must be a mapping, got {type(payload)!r}.")
    return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        payload = yaml.safe_load(file_handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML at {path} must contain a top-level mapping.")
    return payload


def _normalize_defaults(defaults_payload: Any) -> dict[str, str]:
    if defaults_payload is None:
        return {}

    if isinstance(defaults_payload, dict):
        return {str(key): str(value) for key, value in defaults_payload.items()}

    if isinstance(defaults_payload, list):
        defaults_map: dict[str, str] = {}
        for item in defaults_payload:
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                defaults_map[str(key)] = str(value)
        return defaults_map

    raise TypeError(f"`defaults` must be a mapping or list of mappings, got {type(defaults_payload)!r}.")


def _load_train_payload(task_name: str, config_dir: Path) -> dict[str, Any]:
    base_train_path = config_dir / "train.yaml"
    if not base_train_path.exists():
        raise FileNotFoundError(f"Missing base train config: {base_train_path}")

    base_train = _read_yaml(base_train_path)

    task_override_path = config_dir / "train" / f"{_task_slug(task_name)}.yaml"
    task_override = _read_yaml(task_override_path) if task_override_path.exists() else {}

    merged_train = _deep_merge(base_train, task_override)

    defaults_map = _normalize_defaults(merged_train.get("defaults"))
    defaults_payload: dict[str, Any] = {}
    for section_key, rel_path in defaults_map.items():
        default_path = config_dir / rel_path
        if not default_path.exists():
            raise FileNotFoundError(f"Config defaults path for '{section_key}' does not exist: {default_path}")
        defaults_payload[section_key] = _read_yaml(default_path)

    merged_train.pop("defaults", None)
    return _deep_merge(defaults_payload, merged_train)


def _parse_torch_dtype(value: Any, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        lookup = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "bfloat16": torch.bfloat16,
        }
        if value not in lookup:
            raise ValueError(f"Unsupported dtype '{value}'. Expected one of: {', '.join(sorted(lookup.keys()))}.")
        return lookup[value]
    raise TypeError(f"`store_dtype` must be str or torch.dtype, got {type(value)!r}.")


def _parse_feature_buffer_cfg(section: dict[str, Any], default_cfg: FeatureBufCfg) -> FeatureBufCfg:
    stale_key = "store_discounted_feature_returns"
    if stale_key in section:
        raise ValueError(
            f"`runner.*_buffer.{stale_key}` is no longer supported. "
            "FeatureTrajectoryBuffer always stores episode features."
        )

    return FeatureBufCfg(
        capacity_steps=int(section.get("capacity_steps", default_cfg.capacity_steps)),
        store_device=str(section.get("store_device", default_cfg.store_device)),
        store_dtype=_parse_torch_dtype(section.get("store_dtype"), default_cfg.store_dtype),
        min_ep_len=int(section.get("min_ep_len", default_cfg.min_ep_len)),
        sample_weighted_by_length=bool(section.get("sample_weighted_by_length", default_cfg.sample_weighted_by_length)),
    )


def _parse_runner_cfg(section: dict[str, Any]) -> IrlRunnerCfg:
    stale_key = "reward_update_interval"
    if stale_key in section:
        raise ValueError(
            "`runner.reward_update_interval` is no longer supported. "
            "Use `runner.reward_updates_per_cycle` instead."
        )

    defaults = IrlRunnerCfg()
    imitator_buffer = _parse_feature_buffer_cfg(
        _to_mapping(section.get("imitator_buffer"), section_name="runner.imitator_buffer"),
        defaults.imitator_buffer,
    )
    expert_buffer = _parse_feature_buffer_cfg(
        _to_mapping(section.get("expert_buffer"), section_name="runner.expert_buffer"),
        defaults.expert_buffer,
    )

    return IrlRunnerCfg(
        num_steps_per_env_rl=int(section.get("num_steps_per_env_rl", defaults.num_steps_per_env_rl)),
        save_interval=int(section.get("save_interval", defaults.save_interval)),
        policy_updates_per_cycle=int(section.get("policy_updates_per_cycle", defaults.policy_updates_per_cycle)),
        reward_updates_per_cycle=int(section.get("reward_updates_per_cycle", defaults.reward_updates_per_cycle)),
        imitator_buffer=imitator_buffer,
        expert_buffer=expert_buffer,
        expert_num_envs=int(section.get("expert_num_envs", defaults.expert_num_envs)),
    )


def _parse_irl_cfg(section: dict[str, Any]) -> IRLCfg:
    stale_key = "reward_gradient_mode"
    if stale_key in section:
        raise ValueError(
            "`irl.reward_gradient_mode` has been removed. "
            "Use `irl.discount_gamma` and `irl.normalize_returns_by_episode_length`."
        )

    defaults = IRLCfg()
    reward_lr = section.get("reward_learning_rate", defaults.reward_learning_rate)
    discount_gamma_raw = section.get("discount_gamma", defaults.discount_gamma)
    discount_gamma = float(discount_gamma_raw) if discount_gamma_raw is not None else None
    if discount_gamma is not None and not (0.0 < discount_gamma <= 1.0):
        raise ValueError(f"`irl.discount_gamma` must be in (0, 1], got {discount_gamma}.")
    return IRLCfg(
        expert_data_path=str(section.get("expert_data_path", defaults.expert_data_path)),
        batch_size=int(section.get("batch_size", defaults.batch_size)),
        num_learning_epochs=int(section.get("num_learning_epochs", defaults.num_learning_epochs)),
        weight_decay=float(section.get("weight_decay", defaults.weight_decay)),
        max_grad_norm=float(section.get("max_grad_norm", defaults.max_grad_norm)),
        reward_loss_coef=float(section.get("reward_loss_coef", defaults.reward_loss_coef)),
        reward_learning_rate=(float(reward_lr) if reward_lr is not None else None),
        discount_gamma=discount_gamma,
        normalize_returns_by_episode_length=bool(
            section.get(
                "normalize_returns_by_episode_length",
                defaults.normalize_returns_by_episode_length,
            )
        ),
    )


def _parse_feature_map_cfg(section: dict[str, Any]) -> ManagerBasedFeatureCfg:
    ignored_terms = section.get("ignored_reward_terms", [])
    if ignored_terms is None:
        ignored_terms = []
    if not isinstance(ignored_terms, (list, tuple, set)):
        raise TypeError("`feature_map.ignored_reward_terms` must be a sequence of term names.")
    return ManagerBasedFeatureCfg(ignored_reward_terms=set(str(item) for item in ignored_terms))


def _parse_actor_critic_cfg(section: dict[str, Any]) -> ActorCriticCfg:
    defaults = ActorCriticCfg()
    return ActorCriticCfg(
        actor_hidden_dims=_to_tuple_ints(section.get("actor_hidden_dims"), defaults.actor_hidden_dims),
        critic_hidden_dims=_to_tuple_ints(section.get("critic_hidden_dims"), defaults.critic_hidden_dims),
        activation=str(section.get("activation", defaults.activation)),
        init_noise_std=float(section.get("init_noise_std", defaults.init_noise_std)),
        noise_std_type=str(section.get("noise_std_type", defaults.noise_std_type)),
        state_dependent_std=bool(section.get("state_dependent_std", defaults.state_dependent_std)),
        actor_obs_normalization=bool(section.get("actor_obs_normalization", defaults.actor_obs_normalization)),
        critic_obs_normalization=bool(section.get("critic_obs_normalization", defaults.critic_obs_normalization)),
    )


def _parse_ppo_cfg(section: dict[str, Any]) -> PpoAlgoCfg:
    defaults = PpoAlgoCfg()
    return PpoAlgoCfg(
        learning_rate=float(section.get("learning_rate", defaults.learning_rate)),
        gamma=float(section.get("gamma", defaults.gamma)),
        lam=float(section.get("lam", defaults.lam)),
        num_learning_epochs=int(section.get("num_learning_epochs", defaults.num_learning_epochs)),
        num_mini_batches=int(section.get("num_mini_batches", defaults.num_mini_batches)),
        schedule=str(section.get("schedule", defaults.schedule)),
        entropy_coef=float(section.get("entropy_coef", defaults.entropy_coef)),
        desired_kl=float(section.get("desired_kl", defaults.desired_kl)),
        max_grad_norm=float(section.get("max_grad_norm", defaults.max_grad_norm)),
        value_loss_coef=float(section.get("value_loss_coef", defaults.value_loss_coef)),
        use_clipped_value_loss=bool(section.get("use_clipped_value_loss", defaults.use_clipped_value_loss)),
        clip_param=float(section.get("clip_param", defaults.clip_param)),
        normalize_advantage_per_mini_batch=bool(
            section.get("normalize_advantage_per_mini_batch", defaults.normalize_advantage_per_mini_batch)
        ),
        rnd_cfg=section.get("rnd_cfg", defaults.rnd_cfg),
        symmetry_cfg=section.get("symmetry_cfg", defaults.symmetry_cfg),
    )


def _parse_reward_cfg(section: dict[str, Any]) -> RewardModelCfg:
    if "reward_hidden_dims" in section or "reward_is_linear" in section:
        raise ValueError(
            "Reward config keys `reward_hidden_dims` and `reward_is_linear` are no longer supported. "
            "Use `hidden_dims` and `is_linear` under `reward:`."
        )

    reward_type = str(section.get("type", "dense_mlp"))
    if reward_type not in {"dense_mlp", "dense"}:
        raise ValueError(f"Unsupported reward type '{reward_type}'. Expected 'dense_mlp' or 'dense'.")

    defaults = RewardModelCfg(num_features=1)
    return RewardModelCfg(
        num_features=int(section.get("num_features", defaults.num_features)),
        hidden_dims=_to_tuple_ints(section.get("hidden_dims"), defaults.hidden_dims),
        is_linear=bool(section.get("is_linear", defaults.is_linear)),
        activation=str(section.get("activation", defaults.activation)),
    )


def load_train_cfg(
    *,
    task_name: str,
    args_cli: argparse.Namespace,
    config_dir: Path | None = None,
) -> TrainCfg:
    config_root = config_dir or (_repo_root() / "configs")
    payload = _load_train_payload(task_name=task_name, config_dir=config_root)

    env_section = _to_mapping(payload.get("env"), section_name="env")
    resolved_task_name = str(task_name or env_section.get("name") or "")
    if resolved_task_name == "":
        raise ValueError("Task name is required. Provide `--task` or set `env.name` in train config.")

    default_experiment_name = _task_slug(resolved_task_name)
    cfg = TrainCfg(
        experiment_name=str(payload.get("experiment_name", default_experiment_name)),
        run_name=str(payload.get("run_name", "")),
        logger=str(payload.get("logger", "tensorboard")),
        seed=int(payload.get("seed", 42)),
        max_iterations=int(payload.get("max_iterations", 1500)),
        device=str(env_section.get("device", "cuda:0")),
        env=EnvCfg(
            name=resolved_task_name,
            device=str(env_section.get("device", "cuda:0")),
            num_envs=(int(env_section["num_envs"]) if env_section.get("num_envs") is not None else None),
        ),
    )

    cfg.feature_map = _parse_feature_map_cfg(_to_mapping(payload.get("feature_map"), section_name="feature_map"))
    cfg.irl = _parse_irl_cfg(_to_mapping(payload.get("irl"), section_name="irl"))
    cfg.runner = _parse_runner_cfg(_to_mapping(payload.get("runner"), section_name="runner"))
    cfg.reward = _parse_reward_cfg(_to_mapping(payload.get("reward"), section_name="reward"))
    cfg.actor_critic = _parse_actor_critic_cfg(_to_mapping(payload.get("policy"), section_name="policy"))
    cfg.rl_algorithm = _parse_ppo_cfg(_to_mapping(payload.get("algo"), section_name="algo"))

    # CLI overrides
    if getattr(args_cli, "seed", None) is not None:
        cfg.seed = int(args_cli.seed)
    if getattr(args_cli, "max_iterations", None) is not None:
        cfg.max_iterations = int(args_cli.max_iterations)
    if int(cfg.max_iterations) <= 0:
        raise ValueError(
            f"`max_iterations` must be > 0, got {int(cfg.max_iterations)}. "
            "Set a positive value in configs/train.yaml or pass --max_iterations."
        )

    cli_device = getattr(args_cli, "device", None)
    if cli_device is not None:
        cfg.device = str(cli_device)
        cfg.env.device = str(cli_device)

    cli_num_envs = getattr(args_cli, "num_envs", None)
    if cli_num_envs is not None:
        cfg.env.num_envs = int(cli_num_envs)

    if getattr(args_cli, "resume", None) is not None:
        cfg.resume = bool(args_cli.resume)
    if getattr(args_cli, "load_run", None) is not None:
        cfg.load_run = str(args_cli.load_run)
    if getattr(args_cli, "checkpoint", None) is not None:
        cfg.load_checkpoint = str(args_cli.checkpoint)
    if getattr(args_cli, "run_name", None) is not None:
        cfg.run_name = str(args_cli.run_name)
    if getattr(args_cli, "experiment_name", None) is not None:
        cfg.experiment_name = str(args_cli.experiment_name)
    if getattr(args_cli, "logger", None) is not None:
        cfg.logger = str(args_cli.logger)

    log_project_name = getattr(args_cli, "log_project_name", None)
    if cfg.logger in {"wandb", "neptune"} and log_project_name:
        cfg.wandb_project = str(log_project_name)
        cfg.neptune_project = str(log_project_name)

    expert_data_path = getattr(args_cli, "expert_data_path", None)
    if expert_data_path:
        cfg.irl = replace(cfg.irl, expert_data_path=str(expert_data_path))
    irl_discount_gamma = getattr(args_cli, "irl_discount_gamma", None)
    if irl_discount_gamma is not None:
        irl_discount_gamma = float(irl_discount_gamma)
        if not (0.0 < irl_discount_gamma <= 1.0):
            raise ValueError(f"`--irl_discount_gamma` must be in (0, 1], got {irl_discount_gamma}.")
        cfg.irl = replace(cfg.irl, discount_gamma=irl_discount_gamma)

    return cfg


def _build_feature_map(
    cfg: ManagerBasedFeatureCfg,
    device: str | torch.device,
) -> Callable[[Any], torch.Tensor]:
    def _feature_map(env: Any) -> torch.Tensor:
        return manager_based_reward_features(
            env=env,
            ignored_reward_terms=cfg.ignored_reward_terms,
            device=device,
        )

    return _feature_map


def _feature_dim_from_feature_map(feature_map: Callable[[Any], torch.Tensor], env: Any) -> int:
    with torch.no_grad():
        features = feature_map(env)
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 2:
        raise ValueError(f"Feature map must return shape [N, D], got {tuple(features.shape)}")
    return int(features.shape[1])


def _torch_load_payload(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        pass
    except Exception:
        pass
    return torch.load(path, map_location="cpu")


def _tensor_to_feature_episodes(payload: torch.Tensor) -> list[torch.Tensor]:
    if payload.ndim == 2:
        return [payload]
    if payload.ndim == 3:
        return [payload[i] for i in range(payload.shape[0])]
    raise ValueError(
        f"Expected tensor payload with shape [T, D] or [E, T, D], got {tuple(payload.shape)}"
    )


def _extract_feature_episodes_from_torch_payload(payload: Any) -> list[torch.Tensor]:
    if isinstance(payload, dict):
        keys = ("episodes", "features", "expert_episodes")
        selected_key = next((key for key in keys if key in payload), None)
        if selected_key is None:
            raise ValueError(
                "Torch payload must contain one of keys: 'episodes', 'features', 'expert_episodes'."
            )
        payload = payload[selected_key]

    if isinstance(payload, torch.Tensor):
        return _tensor_to_feature_episodes(payload)

    if not isinstance(payload, (list, tuple)):
        raise TypeError(f"Expected list/tuple/tensor feature payload, got {type(payload)!r}")

    episodes: list[torch.Tensor] = []
    for episode in payload:
        episode_tensor = episode if isinstance(episode, torch.Tensor) else torch.as_tensor(episode)
        episodes.extend(_tensor_to_feature_episodes(episode_tensor))
    return episodes


def _extract_feature_episodes_from_hdf5(path: str) -> list[torch.Tensor]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading .h5/.hdf5 expert data requires `h5py`. Install it in your environment."
        ) from exc

    def _episode_from_named_features(feature_group: Any, *, demo_key: str) -> torch.Tensor:
        if not hasattr(feature_group, "keys"):
            raise ValueError(
                f"Demo '{demo_key}' uses legacy feature format. Expected group 'features/<feature_name>'. "
                "Regenerate demos with the simplified recorder."
            )

        feature_names = list(feature_group.keys())
        if len(feature_names) == 0:
            raise ValueError(f"Demo '{demo_key}' has empty 'features' group.")

        columns: list[torch.Tensor] = []
        expected_steps: int | None = None
        for feature_name in feature_names:
            feature_node = feature_group[feature_name]
            if hasattr(feature_node, "keys"):
                raise ValueError(
                    f"Demo '{demo_key}' feature '{feature_name}' must be a dataset of shape [T] or [T,1]."
                )

            values = torch.as_tensor(feature_node[...])
            if values.ndim == 2 and int(values.shape[1]) == 1:
                values = rearrange(values, "t 1 -> t")
            elif values.ndim != 1:
                raise ValueError(
                    f"Demo '{demo_key}' feature '{feature_name}' must have shape [T] or [T,1], "
                    f"got {tuple(values.shape)}."
                )

            num_steps = int(values.shape[0])
            if expected_steps is None:
                if num_steps <= 0:
                    raise ValueError(f"Demo '{demo_key}' has empty feature sequence '{feature_name}'.")
                expected_steps = num_steps
            elif num_steps != expected_steps:
                raise ValueError(
                    f"Demo '{demo_key}' feature length mismatch: expected {expected_steps}, "
                    f"feature '{feature_name}' has length {num_steps}."
                )

            columns.append(rearrange(values, "t -> t 1"))

        return torch.cat(columns, dim=1)

    episodes: list[torch.Tensor] = []
    with h5py.File(path, "r") as file_handle:
        data_group = file_handle.get("data", None)
        if data_group is None or not hasattr(data_group, "keys"):
            raise ValueError(
                "HDF5 payload must contain demos under 'data/demo_*/features/<feature_name>'. "
                "Regenerate demos with the simplified recorder."
            )

        for demo_key in sorted(data_group.keys()):
            demo_group = data_group[demo_key]
            if not hasattr(demo_group, "keys"):
                raise ValueError(f"Demo '{demo_key}' must be a group containing 'features'.")
            if "features" not in demo_group:
                raise ValueError(f"Demo '{demo_key}' is missing required group 'features'.")

            episodes.append(_episode_from_named_features(demo_group["features"], demo_key=demo_key))

    if len(episodes) == 0:
        raise ValueError("No demos found under 'data/*' in HDF5 payload.")

    return episodes


def _validate_feature_episodes(
    episodes: list[torch.Tensor],
    expected_feature_dim: int,
) -> list[torch.Tensor]:
    if len(episodes) == 0:
        raise ValueError("No expert feature episodes found.")

    validated: list[torch.Tensor] = []
    for episode_idx, episode in enumerate(episodes):
        episode_tensor = episode if isinstance(episode, torch.Tensor) else torch.as_tensor(episode)
        if episode_tensor.ndim != 2:
            raise ValueError(
                f"Episode {episode_idx} must have shape [T, D], got {tuple(episode_tensor.shape)}"
            )
        if episode_tensor.shape[0] <= 0:
            raise ValueError(f"Episode {episode_idx} is empty. Expected at least one timestep.")
        if int(episode_tensor.shape[1]) != int(expected_feature_dim):
            raise ValueError(
                f"Episode {episode_idx} feature dim mismatch: "
                f"expected {expected_feature_dim}, got {int(episode_tensor.shape[1])}."
            )
        validated.append(episode_tensor)

    return validated


def load_feature_episodes(path: str, expected_feature_dim: int) -> list[torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert data path does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in {".h5", ".hdf5"}:
        episodes = _extract_feature_episodes_from_hdf5(path)
    elif ext in {".pt", ".pth"}:
        payload = _torch_load_payload(path)
        episodes = _extract_feature_episodes_from_torch_payload(payload)
    else:
        raise ValueError(
            f"Unsupported expert data extension '{ext}'. Expected one of: .pt, .pth, .h5, .hdf5"
        )

    return _validate_feature_episodes(episodes, expected_feature_dim=expected_feature_dim)


def _make_expert_buffer_loader(
    expert_data_path: str,
    *,
    expected_feature_dim: int,
) -> Callable[[Any], None]:
    resolved_path = os.path.abspath(expert_data_path)

    def _loader(buffer: Any) -> None:
        if not hasattr(buffer, "ctx"):
            raise AttributeError("Expert buffer must expose runtime context via `buffer.ctx`.")
        if int(buffer.ctx.num_envs) != 1:
            raise ValueError(
                f"Expert loader requires buffer with num_envs == 1, got {int(buffer.ctx.num_envs)}."
            )

        episodes = load_feature_episodes(resolved_path, expected_feature_dim=expected_feature_dim)
        for episode in episodes:
            buffer.add_episode(episode)

    return _loader


def _build_log_dir(experiment_name: str, run_name: str) -> tuple[str, str]:
    log_root_path = os.path.abspath(os.path.join("logs", "irl", experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        log_dir += f"_{run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return log_root_path, log_dir


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


def _summarize_env_cfg(env_cfg: Any, train_cfg: TrainCfg) -> dict[str, Any]:
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


def _dump_run_configs(log_dir: str, env_cfg: Any, train_cfg: TrainCfg) -> None:
    serializable_train_cfg = _to_serializable(train_cfg)
    params_dir = os.path.join(log_dir, "params")
    _dump_yaml(os.path.join(params_dir, "train.yaml"), serializable_train_cfg)
    env_summary = _summarize_env_cfg(env_cfg, train_cfg)
    _dump_yaml(os.path.join(params_dir, "env.yaml"), env_summary)


def _rename_video_files_to_iteration_index(video_dir: str, steps_per_iteration: int) -> None:
    if steps_per_iteration <= 0:
        raise ValueError(f"`steps_per_iteration` must be > 0, got {steps_per_iteration}.")
    if not os.path.isdir(video_dir):
        return

    step_name_pattern = re.compile(r"^(?P<prefix>.+)-step-(?P<step>\d+)(?P<ext>\.mp4)$")
    for file_name in sorted(os.listdir(video_dir)):
        match = step_name_pattern.match(file_name)
        if match is None:
            continue

        step_idx = int(match.group("step"))
        iter_idx = step_idx // steps_per_iteration
        new_name = f"{match.group('prefix')}-iter-{iter_idx}{match.group('ext')}"

        source_path = os.path.join(video_dir, file_name)
        target_path = os.path.join(video_dir, new_name)
        if source_path == target_path:
            continue
        os.replace(source_path, target_path)


def _get_initial_obs(
    env: Any,
    resolve_obs_groups_fn: Callable[..., dict[str, list[str]]],
) -> tuple[Any, dict[str, list[str]]]:
    obs = env.get_observations()
    if not hasattr(obs, "keys"):
        raise TypeError("RslRlVecEnvWrapper.get_observations() must return TensorDict-like observations.")
    obs_groups = resolve_obs_groups_fn(obs, {}, default_sets=["critic"])
    return obs, obs_groups


def _load_app_launcher_cls() -> Any:
    from isaaclab.app import AppLauncher

    return AppLauncher


def _load_runtime_deps(app_launcher_cls: Any) -> RuntimeDeps:

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    from rsl_rl.utils import resolve_obs_groups

    return RuntimeDeps(
        AppLauncher=app_launcher_cls,
        gym=gym,
        parse_env_cfg=parse_env_cfg,
        get_checkpoint_path=get_checkpoint_path,
        RslRlVecEnvWrapper=RslRlVecEnvWrapper,
        PPO=PPO,
        ActorCritic=ActorCritic,
        resolve_obs_groups=resolve_obs_groups,
    )


def _build_arg_parser(app_launcher_cls: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an IRL agent with PPO + feature-buffer IRL.")

    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded videos (steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recordings (steps).")
    parser.add_argument(
        "--video_interval_iterations",
        type=int,
        default=None,
        help="Interval between recordings (learning iterations). Overrides --video_interval.",
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument("--task", type=str, default=None, help="Task name.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Environment / training seed.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
    parser.add_argument(
        "--irl_discount_gamma",
        type=float,
        default=None,
        help="Optional discount gamma override for IRL reward updates.",
    )
    parser.add_argument(
        "--expert_data_path",
        type=str,
        default=None,
        help="Path to expert feature episodes (.pt/.pth/.h5/.hdf5).",
    )

    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name for logging.")
    parser.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
    parser.add_argument("--resume", action="store_true", default=None, help="Resume from a previous checkpoint.")
    parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    parser.add_argument(
        "--logger",
        type=str,
        choices={"wandb", "tensorboard", "neptune"},
        default=None,
        help="Logger backend.",
    )
    parser.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Project name for wandb/neptune loggers.",
    )

    app_launcher_cls.add_app_launcher_args(parser)
    return parser


def main(argv: list[str] | None = None, deps: RuntimeDeps | None = None) -> None:
    runtime_deps = deps
    if runtime_deps is None:
        app_launcher_cls = _load_app_launcher_cls()
    else:
        app_launcher_cls = runtime_deps.AppLauncher

    parser = _build_arg_parser(app_launcher_cls)
    args_cli = parser.parse_args(argv)
    if args_cli.video:
        args_cli.enable_cameras = True

    app_launcher = app_launcher_cls(args_cli)
    simulation_app = app_launcher.app
    env = None
    video_dir_for_iteration_names: str | None = None
    video_steps_per_iteration: int | None = None

    if runtime_deps is None:
        # Isaac/Omniverse runtime deps must be imported only after SimulationApp creation.
        runtime_deps = _load_runtime_deps(app_launcher_cls=app_launcher_cls)

    try:
        requested_task_name = args_cli.task
        if requested_task_name is None:
            raise ValueError("`--task` is required.")

        train_cfg = load_train_cfg(task_name=requested_task_name, args_cli=args_cli)

        env_cfg = runtime_deps.parse_env_cfg(
            requested_task_name,
            device=train_cfg.env.device,
            num_envs=train_cfg.env.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )

        log_root_path, log_dir = _build_log_dir(
            experiment_name=train_cfg.experiment_name,
            run_name=train_cfg.run_name,
        )

        env = runtime_deps.gym.make(
            requested_task_name,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None,
        )

        if args_cli.video:
            if args_cli.video_interval_iterations is not None:
                if int(args_cli.video_interval_iterations) <= 0:
                    raise ValueError(
                        f"`--video_interval_iterations` must be > 0, got {args_cli.video_interval_iterations}."
                    )
                video_interval_steps = int(args_cli.video_interval_iterations) * int(train_cfg.runner.num_steps_per_env_rl)
                video_dir_for_iteration_names = os.path.join(log_dir, "videos", "train")
                video_steps_per_iteration = int(train_cfg.runner.num_steps_per_env_rl)
            else:
                if int(args_cli.video_interval) <= 0:
                    raise ValueError(f"`--video_interval` must be > 0, got {args_cli.video_interval}.")
                video_interval_steps = int(args_cli.video_interval)

            def _video_step_trigger(step_idx: int) -> bool:
                return step_idx % video_interval_steps == 0

            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": _video_step_trigger,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = runtime_deps.gym.wrappers.RecordVideo(env, **video_kwargs)

        env = runtime_deps.RslRlVecEnvWrapper(env)
        if hasattr(env, "seed"):
            env.seed(int(train_cfg.seed))

        obs, obs_groups = _get_initial_obs(env, runtime_deps.resolve_obs_groups)
        obs = obs.to(train_cfg.device)
        actor_critic = runtime_deps.ActorCritic(
            obs,
            obs_groups,
            env.num_actions,
            **asdict(train_cfg.actor_critic),
        ).to(train_cfg.device)
        rl_alg = runtime_deps.PPO(
            policy=actor_critic,
            device=train_cfg.device,
            **asdict(train_cfg.rl_algorithm),
        )

        feature_map = _build_feature_map(train_cfg.feature_map, device=train_cfg.device)
        feature_dim = _feature_dim_from_feature_map(feature_map, env)
        runtime_ctx = RuntimeContext(
            num_envs=int(env.num_envs),
            feature_dim=feature_dim,
            device=train_cfg.device,
        )
        reward_model = RewardModel(replace(train_cfg.reward, num_features=runtime_ctx.feature_dim)).to(train_cfg.device)

        irl_alg = IRL(
            rl_alg=rl_alg,
            reward=reward_model,
            gamma=float(train_cfg.rl_algorithm.gamma),
            cfg=train_cfg.irl,
            env=env,
            feature_map=feature_map,
            device=train_cfg.device,
        )

        expert_loader = None
        if train_cfg.irl.expert_data_path:
            expert_loader = _make_expert_buffer_loader(
                train_cfg.irl.expert_data_path,
                expected_feature_dim=runtime_ctx.feature_dim,
            )

        runner = IrlRunner(
            env=env,
            rl_alg=rl_alg,
            irl_alg=irl_alg,
            feature_map=feature_map,
            runner_cfg=train_cfg.runner,
            runtime_ctx=runtime_ctx,
            log_dir=log_dir,
            device=train_cfg.device,
            expert_buffer_loader=expert_loader,
            reward_env_wrapper_factory=None,
        )

        if train_cfg.resume:
            resume_path = runtime_deps.get_checkpoint_path(
                log_root_path,
                train_cfg.load_run,
                train_cfg.load_checkpoint,
            )
            runner.load(resume_path, load_optimizer=True)

        _dump_run_configs(log_dir=log_dir, env_cfg=env_cfg, train_cfg=train_cfg)
        runner.learn(
            num_learning_iterations=int(train_cfg.max_iterations),
            init_at_random_ep_len=True,
        )

    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception as exc:
                print(f"[WARN] Failed to close environment cleanly: {exc}")
        if video_dir_for_iteration_names is not None and video_steps_per_iteration is not None:
            try:
                _rename_video_files_to_iteration_index(
                    video_dir=video_dir_for_iteration_names,
                    steps_per_iteration=video_steps_per_iteration,
                )
            except Exception as exc:
                print(f"[WARN] Failed to rename video files with iteration indices: {exc}")
        simulation_app.close()


if __name__ == "__main__":
    main()
