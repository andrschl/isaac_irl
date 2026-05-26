"""Config dataclasses, YAML+CLI parsers, and run-config dumping for train_irl.

Module-level imports are kept light (no Isaac Lab / rsl_rl) so this file can
be loaded in tests without the SimulationApp dance.

Public surface:
    ActorCriticCfg, PpoAlgoCfg, EnvCfg, TrainCfg,
    load_train_cfg, _dump_run_configs
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

# Ensure src/ is importable when loaded standalone via importlib in tests.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from algorithms import IRLCfg
from algorithms.bc import BCCfg
from reward_features.manager_based import ManagerBasedFeatureCfg
from reward_model import RewardModelCfg
from runner import IrlRunnerCfg
from storage.feature_storage import FeatureBufCfg


__all__ = [
    "ActorCriticCfg",
    "PpoAlgoCfg",
    "EnvCfg",
    "TrainCfg",
    "load_train_cfg",
    "_dump_run_configs",
]


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


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
    success_threshold: float = 0.08
    """Object-to-goal distance (m) below which a Lift step counts as success.
    Centralized here so train_irl.py and the recording script agree without
    requiring a CLI flag every run."""
    add_success_bonus: bool = True
    """If True, inject a `success_bonus` reward term (binary object_reached_goal)
    into the env's RewardsCfg before gym.make. The term is exposed as an IRL
    feature (auto-registered in ManagerBasedFeatureCfg.force_include_terms);
    its env-reward contribution is controlled by `success_bonus_weight`."""
    success_bonus_weight: float = 0.0
    """Weight of the injected success_bonus term in the env reward signal.
    Defaults to 0.0 so the term is feature-only (IRL still learns its weight
    via the reward model). Set > 0 to also contribute to the env reward."""


@dataclass(slots=True)
class TrainCfg:
    experiment_name: str = "default_experiment"
    run_name: str = ""
    logger: str = "wandb"
    wandb_project: str = "isaaclab"

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
    bc: BCCfg = field(default_factory=BCCfg)


# ---------------------------------------------------------------------------
# Small YAML / dtype helpers
# ---------------------------------------------------------------------------


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


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        payload = yaml.safe_load(file_handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML at {path} must contain a top-level mapping.")
    return payload


def _load_train_payload(task_name: str, config_dir: Path) -> dict[str, Any]:
    del task_name  # config layout currently shares a single experiment.yaml
    experiment_path = config_dir / "experiment.yaml"
    if not experiment_path.exists():
        raise FileNotFoundError(f"Missing experiment config: {experiment_path}")
    return _read_yaml(experiment_path)


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


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def _parse_feature_buffer_cfg(
    section: dict[str, Any], default_cfg: FeatureBufCfg
) -> FeatureBufCfg:
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
    stale_key_messages = {
        "reward_update_interval": "Use `runner.reward_updates_per_cycle` instead.",
        "bc_eval_interval": "Use `runner.validation_interval` instead.",
        "num_steps_per_env_rl": "Use `runner.steps_per_env_per_cycle` instead.",
        "policy_updates_per_cycle": "Use `runner.rl_updates_per_cycle` instead.",
    }
    for stale_key, hint in stale_key_messages.items():
        if stale_key in section:
            raise ValueError(f"`runner.{stale_key}` is no longer supported. {hint}")

    defaults = IrlRunnerCfg()
    imitator_buffer = _parse_feature_buffer_cfg(
        _to_mapping(section.get("imitator_buffer"), section_name="runner.imitator_buffer"),
        defaults.imitator_buffer,
    )
    expert_buffer = _parse_feature_buffer_cfg(
        _to_mapping(section.get("expert_buffer"), section_name="runner.expert_buffer"),
        defaults.expert_buffer,
    )

    steps_per_env_per_cycle = int(section.get("steps_per_env_per_cycle", defaults.steps_per_env_per_cycle))
    if steps_per_env_per_cycle <= 0:
        raise ValueError(
            "`runner.steps_per_env_per_cycle` must be > 0, "
            f"got {steps_per_env_per_cycle}."
        )
    save_interval = int(section.get("save_interval", defaults.save_interval))
    if save_interval <= 0:
        raise ValueError(f"`runner.save_interval` must be > 0, got {save_interval}.")
    rl_updates_per_cycle = int(section.get("rl_updates_per_cycle", defaults.rl_updates_per_cycle))
    if rl_updates_per_cycle <= 0:
        raise ValueError(
            "`runner.rl_updates_per_cycle` must be > 0, "
            f"got {rl_updates_per_cycle}."
        )
    reward_updates_per_cycle = int(section.get("reward_updates_per_cycle", defaults.reward_updates_per_cycle))
    if reward_updates_per_cycle <= 0:
        raise ValueError(
            "`runner.reward_updates_per_cycle` must be > 0, "
            f"got {reward_updates_per_cycle}."
        )
    expert_num_envs = int(section.get("expert_num_envs", defaults.expert_num_envs))
    if expert_num_envs <= 0:
        raise ValueError(f"`runner.expert_num_envs` must be > 0, got {expert_num_envs}.")

    validation_interval = int(section.get("validation_interval", defaults.validation_interval))
    if validation_interval < 0:
        raise ValueError(f"`runner.validation_interval` must be >= 0, got {validation_interval}.")
    validation_steps_per_env_raw = section.get("validation_steps_per_env", defaults.validation_steps_per_env)
    validation_steps_per_env = None if validation_steps_per_env_raw is None else int(validation_steps_per_env_raw)
    if validation_steps_per_env is not None and validation_steps_per_env < 0:
        raise ValueError(
            "`runner.validation_steps_per_env` must be >= 0 or null, "
            f"got {validation_steps_per_env}."
        )

    imitator_rollout_steps_raw = section.get(
        "imitator_rollout_steps_per_env", defaults.imitator_rollout_steps_per_env
    )
    imitator_rollout_steps_per_env = None if imitator_rollout_steps_raw is None else int(imitator_rollout_steps_raw)
    if imitator_rollout_steps_per_env is not None and imitator_rollout_steps_per_env <= 0:
        raise ValueError(
            "`runner.imitator_rollout_steps_per_env` must be > 0 or null, "
            f"got {imitator_rollout_steps_per_env}."
        )

    return IrlRunnerCfg(
        steps_per_env_per_cycle=steps_per_env_per_cycle,
        save_interval=save_interval,
        rl_updates_per_cycle=rl_updates_per_cycle,
        reward_updates_per_cycle=reward_updates_per_cycle,
        use_learned_reward=bool(section.get("use_learned_reward", defaults.use_learned_reward)),
        imitator_buffer=imitator_buffer,
        expert_buffer=expert_buffer,
        expert_num_envs=expert_num_envs,
        validation_interval=validation_interval,
        validation_steps_per_env=validation_steps_per_env,
        imitator_rollout_steps_per_env=imitator_rollout_steps_per_env,
    )


def _parse_irl_cfg(section: dict[str, Any]) -> IRLCfg:
    stale_key = "reward_gradient_mode"
    if stale_key in section:
        raise ValueError(
            "`irl.reward_gradient_mode` has been removed. "
            "Use `irl.discount_gamma` and `irl.normalize_returns_by_episode_length`."
        )
    stale_key = "use_learned_reward"
    if stale_key in section:
        raise ValueError(
            "`irl.use_learned_reward` is no longer supported. "
            "Use `runner.use_learned_reward` instead."
        )

    defaults = IRLCfg()
    reward_lr = section.get("reward_learning_rate", defaults.reward_learning_rate)
    discount_gamma_raw = section.get("discount_gamma", defaults.discount_gamma)
    discount_gamma = float(discount_gamma_raw) if discount_gamma_raw is not None else None
    if discount_gamma is not None and not (0.0 < discount_gamma <= 1.0):
        raise ValueError(f"`irl.discount_gamma` must be in (0, 1], got {discount_gamma}.")

    expert_num_trajectories_raw = section.get("expert_num_trajectories")
    expert_num_trajectories = (
        int(expert_num_trajectories_raw) if expert_num_trajectories_raw is not None else None
    )
    if expert_num_trajectories is not None and expert_num_trajectories <= 0:
        raise ValueError(
            f"`irl.expert_num_trajectories` must be > 0 when set, got {expert_num_trajectories}."
        )

    return IRLCfg(
        expert_data_path=str(section.get("expert_data_path", defaults.expert_data_path)),
        expert_num_trajectories=expert_num_trajectories,
        expert_subset_strategy=str(
            section.get("expert_subset_strategy", defaults.expert_subset_strategy)
        ),
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
    forced_terms = section.get("force_include_terms", [])
    if forced_terms is None:
        forced_terms = []
    if not isinstance(forced_terms, (list, tuple, set)):
        raise TypeError("`feature_map.force_include_terms` must be a sequence of term names.")
    return ManagerBasedFeatureCfg(
        ignored_reward_terms=set(str(item) for item in ignored_terms),
        force_include_terms=set(str(item) for item in forced_terms),
    )


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


def _parse_bc_cfg(section: dict[str, Any]) -> BCCfg:
    defaults = BCCfg()
    alpha = float(section.get("alpha", defaults.alpha))
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"`bc.alpha` must be in [0, 1], got {alpha}.")
    loss_type = str(section.get("loss_type", defaults.loss_type))
    if loss_type not in {"nll", "mse"}:
        raise ValueError(
            f"`bc.loss_type` must be 'nll' or 'mse', got {loss_type!r}."
        )
    batch_size = int(section.get("batch_size", defaults.batch_size))
    if batch_size <= 0:
        raise ValueError(f"`bc.batch_size` must be > 0, got {batch_size}.")
    val_fraction = float(section.get("val_fraction", defaults.val_fraction))
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"`bc.val_fraction` must be in [0, 1), got {val_fraction}.")
    return BCCfg(
        alpha=alpha, loss_type=loss_type, batch_size=batch_size, val_fraction=val_fraction
    )


def _parse_reward_cfg(section: dict[str, Any]) -> RewardModelCfg:
    if "reward_hidden_dims" in section or "reward_is_linear" in section:
        raise ValueError(
            "Reward config keys `reward_hidden_dims` and `reward_is_linear` are no longer supported. "
            "Use `hidden_dims` and `is_linear` under `reward:`."
        )

    reward_type = str(section.get("type", "dense"))
    if reward_type not in {"dense", "linear"}:
        raise ValueError(
            f"Unsupported reward type '{reward_type}'. Expected 'dense' or 'linear'."
        )

    defaults = RewardModelCfg(num_features=1)
    if reward_type == "linear":
        is_linear = True
    else:
        is_linear = bool(section.get("is_linear", defaults.is_linear))
    linear_projection = str(section.get("linear_projection", defaults.linear_projection))
    if not is_linear and linear_projection != "none":
        raise ValueError(
            "`linear_projection` is only valid when `is_linear` is True. "
            f"Got is_linear={is_linear}, linear_projection={linear_projection!r}."
        )
    return RewardModelCfg(
        num_features=int(section.get("num_features", defaults.num_features)),
        hidden_dims=_to_tuple_ints(section.get("hidden_dims"), defaults.hidden_dims),
        is_linear=is_linear,
        activation=str(section.get("activation", defaults.activation)),
        regularization=str(section.get("regularization", defaults.regularization)),
        regularization_strength=float(section.get("regularization_strength", defaults.regularization_strength)),
        elastic_alpha=float(section.get("elastic_alpha", defaults.elastic_alpha)),
        linear_projection=linear_projection,
        linear_projection_radius=float(section.get("linear_projection_radius", defaults.linear_projection_radius)),
    )


# ---------------------------------------------------------------------------
# Top-level loader (YAML + CLI overrides)
# ---------------------------------------------------------------------------


def load_train_cfg(
    *,
    task_name: str,
    args_cli: argparse.Namespace,
    config_dir: Path | None = None,
) -> TrainCfg:
    config_root = config_dir or (_repo_root() / "configs" / "franka_lift")
    payload = _load_train_payload(task_name=task_name, config_dir=config_root)

    env_section = _to_mapping(payload.get("env"), section_name="env")
    resolved_task_name = str(task_name or env_section.get("name") or "")
    if resolved_task_name == "":
        raise ValueError("Task name is required. Provide `--task` or set `env.name` in train config.")

    default_experiment_name = _task_slug(resolved_task_name)
    env_defaults = EnvCfg()
    cfg = TrainCfg(
        experiment_name=str(payload.get("experiment_name", default_experiment_name)),
        run_name=str(payload.get("run_name", "")),
        logger=str(payload.get("logger", "wandb")),
        seed=int(payload.get("seed", 42)),
        max_iterations=int(payload.get("max_iterations", 1500)),
        device=str(env_section.get("device", "cuda:0")),
        env=EnvCfg(
            name=resolved_task_name,
            device=str(env_section.get("device", "cuda:0")),
            num_envs=(int(env_section["num_envs"]) if env_section.get("num_envs") is not None else None),
            success_threshold=float(env_section.get("success_threshold", env_defaults.success_threshold)),
            add_success_bonus=bool(env_section.get("add_success_bonus", env_defaults.add_success_bonus)),
            success_bonus_weight=float(env_section.get("success_bonus_weight", env_defaults.success_bonus_weight)),
        ),
    )

    cfg.feature_map = _parse_feature_map_cfg(_to_mapping(payload.get("feature_map"), section_name="feature_map"))
    cfg.irl = _parse_irl_cfg(_to_mapping(payload.get("irl"), section_name="irl"))
    cfg.runner = _parse_runner_cfg(_to_mapping(payload.get("runner"), section_name="runner"))
    cfg.reward = _parse_reward_cfg(_to_mapping(payload.get("reward"), section_name="reward"))
    cfg.actor_critic = _parse_actor_critic_cfg(_to_mapping(payload.get("policy"), section_name="policy"))
    cfg.rl_algorithm = _parse_ppo_cfg(_to_mapping(payload.get("algo"), section_name="algo"))
    cfg.bc = _parse_bc_cfg(_to_mapping(payload.get("bc"), section_name="bc"))

    # CLI overrides
    if getattr(args_cli, "seed", None) is not None:
        cfg.seed = int(args_cli.seed)
    if getattr(args_cli, "max_iterations", None) is not None:
        cfg.max_iterations = int(args_cli.max_iterations)
    if int(cfg.max_iterations) <= 0:
        raise ValueError(
            f"`max_iterations` must be > 0, got {int(cfg.max_iterations)}. "
            "Set a positive value in configs/franka_lift/experiment.yaml or pass --max_iterations."
        )

    cli_device = getattr(args_cli, "device", None)
    if cli_device is not None:
        cfg.device = str(cli_device)
        cfg.env.device = str(cli_device)

    cli_num_envs = getattr(args_cli, "num_envs", None)
    if cli_num_envs is not None:
        cfg.env.num_envs = int(cli_num_envs)

    cli_success_threshold = getattr(args_cli, "success_threshold", None)
    if cli_success_threshold is not None:
        cli_success_threshold = float(cli_success_threshold)
        if cli_success_threshold <= 0.0:
            raise ValueError(
                f"`--success_threshold` must be > 0, got {cli_success_threshold}."
            )
        cfg.env.success_threshold = cli_success_threshold

    cli_add_success_bonus = getattr(args_cli, "add_success_bonus", None)
    if cli_add_success_bonus is not None:
        cfg.env.add_success_bonus = bool(cli_add_success_bonus)

    cli_success_bonus_weight = getattr(args_cli, "success_bonus_weight", None)
    if cli_success_bonus_weight is not None:
        cfg.env.success_bonus_weight = float(cli_success_bonus_weight)

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
    if cfg.logger not in {"wandb", "noop"}:
        raise ValueError(f"`logger` must be 'wandb' or 'noop', got {cfg.logger!r}.")

    log_project_name = getattr(args_cli, "log_project_name", None)
    if cfg.logger == "wandb" and log_project_name:
        cfg.wandb_project = str(log_project_name)

    expert_data_path = getattr(args_cli, "expert_data_path", None)
    if expert_data_path:
        cfg.irl = replace(cfg.irl, expert_data_path=str(expert_data_path))
    expert_num_trajectories = getattr(args_cli, "expert_num_trajectories", None)
    if expert_num_trajectories is not None:
        if expert_num_trajectories <= 0:
            raise ValueError(
                f"`--expert_num_trajectories` must be > 0, got {expert_num_trajectories}."
            )
        cfg.irl = replace(cfg.irl, expert_num_trajectories=expert_num_trajectories)
    expert_subset_strategy = getattr(args_cli, "expert_subset_strategy", None)
    if expert_subset_strategy is not None:
        cfg.irl = replace(cfg.irl, expert_subset_strategy=str(expert_subset_strategy))
    irl_discount_gamma = getattr(args_cli, "irl_discount_gamma", None)
    if irl_discount_gamma is not None:
        irl_discount_gamma = float(irl_discount_gamma)
        if not (0.0 < irl_discount_gamma <= 1.0):
            raise ValueError(
                f"`--irl_discount_gamma` must be in (0, 1], got {irl_discount_gamma}."
            )
        cfg.irl = replace(cfg.irl, discount_gamma=irl_discount_gamma)

    use_learned_reward = getattr(args_cli, "use_learned_reward", None)
    if use_learned_reward is not None:
        cfg.runner = replace(cfg.runner, use_learned_reward=bool(use_learned_reward))

    bc_alpha = getattr(args_cli, "bc_alpha", None)
    if bc_alpha is not None:
        bc_alpha = float(bc_alpha)
        if not (0.0 <= bc_alpha <= 1.0):
            raise ValueError(f"`--bc_alpha` must be in [0, 1], got {bc_alpha}.")
        cfg.bc = replace(cfg.bc, alpha=bc_alpha)
    bc_loss_type = getattr(args_cli, "bc_loss_type", None)
    if bc_loss_type is not None:
        cfg.bc = replace(cfg.bc, loss_type=str(bc_loss_type))

    learning_rate = getattr(args_cli, "learning_rate", None)
    if learning_rate is not None:
        learning_rate = float(learning_rate)
        if learning_rate <= 0.0:
            raise ValueError(f"`--learning_rate` must be > 0, got {learning_rate}.")
        cfg.rl_algorithm = replace(cfg.rl_algorithm, learning_rate=learning_rate)

    reward_learning_rate = getattr(args_cli, "reward_learning_rate", None)
    if reward_learning_rate is not None:
        reward_learning_rate = float(reward_learning_rate)
        if reward_learning_rate <= 0.0:
            raise ValueError(
                f"`--reward_learning_rate` must be > 0, got {reward_learning_rate}."
            )
        cfg.irl = replace(cfg.irl, reward_learning_rate=reward_learning_rate)

    reward_regularization_strength = getattr(
        args_cli, "reward_regularization_strength", None
    )
    if reward_regularization_strength is not None:
        reward_regularization_strength = float(reward_regularization_strength)
        if reward_regularization_strength < 0.0:
            raise ValueError(
                f"`--reward_regularization_strength` must be >= 0, got {reward_regularization_strength}."
            )
        cfg.reward = replace(
            cfg.reward, regularization_strength=reward_regularization_strength
        )

    reward_updates_per_cycle = getattr(args_cli, "reward_updates_per_cycle", None)
    if reward_updates_per_cycle is not None:
        reward_updates_per_cycle = int(reward_updates_per_cycle)
        if reward_updates_per_cycle < 1:
            raise ValueError(
                f"`--reward_updates_per_cycle` must be >= 1, got {reward_updates_per_cycle}."
            )
        cfg.runner = replace(
            cfg.runner, reward_updates_per_cycle=reward_updates_per_cycle
        )

    rl_updates_per_cycle = getattr(args_cli, "rl_updates_per_cycle", None)
    if rl_updates_per_cycle is not None:
        rl_updates_per_cycle = int(rl_updates_per_cycle)
        if rl_updates_per_cycle < 1:
            raise ValueError(
                f"`--rl_updates_per_cycle` must be >= 1, got {rl_updates_per_cycle}."
            )
        cfg.runner = replace(cfg.runner, rl_updates_per_cycle=rl_updates_per_cycle)

    validation_interval = getattr(args_cli, "validation_interval", None)
    if validation_interval is not None:
        validation_interval = int(validation_interval)
        if validation_interval < 0:
            raise ValueError(
                f"`--validation_interval` must be >= 0, got {validation_interval}."
            )
        cfg.runner = replace(cfg.runner, validation_interval=validation_interval)

    return cfg


# ---------------------------------------------------------------------------
# Run-config dump (used by main() to write logs/.../params/experiment.yaml)
# ---------------------------------------------------------------------------


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


def _extract_ground_truth_reward_weights(env: Any) -> dict[str, float]:
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


def _dump_run_configs(
    log_dir: str,
    env_cfg: Any,
    train_cfg: TrainCfg,
    env: Any | None = None,
) -> None:
    serializable_train_cfg = _to_serializable(train_cfg)
    params_dir = os.path.join(log_dir, "params")
    env_summary = _summarize_env_cfg(env_cfg, train_cfg)
    experiment_cfg = {**serializable_train_cfg, "env": env_summary}
    _dump_yaml(os.path.join(params_dir, "experiment.yaml"), experiment_cfg)
    if env is not None:
        ground_truth = _extract_ground_truth_reward_weights(env)
        if ground_truth:
            _dump_yaml(
                os.path.join(params_dir, "ground_truth_reward_weights.yaml"),
                ground_truth,
            )
            weights_str = "  ".join(f"{k}={v:.4f}" for k, v in ground_truth.items())
            print(f"[INFO] Ground truth reward weights: {weights_str}")
