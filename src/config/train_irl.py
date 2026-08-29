"""YAML + CLI parsing for the IRL training pipeline.

Pure config: no Isaac/Omniverse imports, no env construction, no runtime
side effects. The training script (`scripts/irl/train_irl.py`) calls
:func:`load_train_cfg` once and passes the resulting :class:`TrainCfg`
into the rest of the wiring.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import torch
import yaml

from algorithms import IRLCfg
from algorithms.bc import BCCfg
from config.off_policy import (
    DirectQCfg,
    OffPolicyTrainCfg,
    PmdCfg,
    SacCfg,
    SuccessorQCfg,
    TanhGaussianPolicyCfg,
    TransitionReplayCfg,
)
from reward_features.manager_based import ManagerBasedFeatureCfg
from reward_model import RewardModelCfg
from runner import IrlRunnerCfg
from storage.feature_storage import FeatureBufCfg


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
    type: ClassVar[str] = "ppo"
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
    init_policy_checkpoint: str | None = None

    seed: int = 42
    max_iterations: int = 1500
    device: str = "cuda:0"
    env: EnvCfg = field(default_factory=EnvCfg)

    feature_map: ManagerBasedFeatureCfg = field(default_factory=ManagerBasedFeatureCfg)
    irl: IRLCfg = field(default_factory=IRLCfg)
    runner: IrlRunnerCfg = field(default_factory=IrlRunnerCfg)
    reward: RewardModelCfg = field(
        default_factory=lambda: RewardModelCfg(num_features=1)
    )
    actor_critic: ActorCriticCfg = field(default_factory=ActorCriticCfg)
    rl_algorithm: PpoAlgoCfg | SacCfg | PmdCfg = field(default_factory=PpoAlgoCfg)
    bc: BCCfg = field(default_factory=BCCfg)


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
        raise TypeError(
            f"Config section '{section_name}' must be a mapping, got {type(payload)!r}."
        )
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        payload = yaml.safe_load(file_handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML at {path} must contain a top-level mapping.")
    return payload


def _load_train_payload(task_name: str, config_dir: Path) -> dict[str, Any]:
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
            raise ValueError(
                f"Unsupported dtype '{value}'. Expected one of: {', '.join(sorted(lookup.keys()))}."
            )
        return lookup[value]
    raise TypeError(f"`store_dtype` must be str or torch.dtype, got {type(value)!r}.")


def _parse_feature_buffer_cfg(
    section: dict[str, Any], default_cfg: FeatureBufCfg
) -> FeatureBufCfg:
    stale_key = "store_discounted_feature_returns"
    if stale_key in section:
        raise ValueError(
            f"`runner.*_buffer.{stale_key}` is no longer supported. "
            "FeatureTrajectoryBuffer always stores episode features."
        )

    raw_store_device = section.get("store_device", default_cfg.store_device)
    store_device = str(raw_store_device) if raw_store_device is not None else None
    return FeatureBufCfg(
        capacity_steps=int(section.get("capacity_steps", default_cfg.capacity_steps)),
        store_device=store_device,
        store_dtype=_parse_torch_dtype(
            section.get("store_dtype"), default_cfg.store_dtype
        ),
        min_ep_len=int(section.get("min_ep_len", default_cfg.min_ep_len)),
        sample_weighted_by_length=bool(
            section.get(
                "sample_weighted_by_length", default_cfg.sample_weighted_by_length
            )
        ),
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
        _to_mapping(
            section.get("imitator_buffer"), section_name="runner.imitator_buffer"
        ),
        defaults.imitator_buffer,
    )
    expert_buffer = _parse_feature_buffer_cfg(
        _to_mapping(section.get("expert_buffer"), section_name="runner.expert_buffer"),
        defaults.expert_buffer,
    )

    steps_per_env_per_cycle = int(
        section.get("steps_per_env_per_cycle", defaults.steps_per_env_per_cycle)
    )
    if steps_per_env_per_cycle <= 0:
        raise ValueError(
            "`runner.steps_per_env_per_cycle` must be > 0, "
            f"got {steps_per_env_per_cycle}."
        )
    save_interval = int(section.get("save_interval", defaults.save_interval))
    if save_interval <= 0:
        raise ValueError(f"`runner.save_interval` must be > 0, got {save_interval}.")
    rl_updates_per_cycle = int(
        section.get("rl_updates_per_cycle", defaults.rl_updates_per_cycle)
    )
    if rl_updates_per_cycle <= 0:
        raise ValueError(
            f"`runner.rl_updates_per_cycle` must be > 0, got {rl_updates_per_cycle}."
        )
    reward_updates_per_cycle = int(
        section.get("reward_updates_per_cycle", defaults.reward_updates_per_cycle)
    )
    if reward_updates_per_cycle <= 0:
        raise ValueError(
            "`runner.reward_updates_per_cycle` must be > 0, "
            f"got {reward_updates_per_cycle}."
        )
    expert_num_envs = int(section.get("expert_num_envs", defaults.expert_num_envs))
    if expert_num_envs <= 0:
        raise ValueError(
            f"`runner.expert_num_envs` must be > 0, got {expert_num_envs}."
        )

    validation_interval = int(
        section.get("validation_interval", defaults.validation_interval)
    )
    if validation_interval < 0:
        raise ValueError(
            f"`runner.validation_interval` must be >= 0, got {validation_interval}."
        )
    validation_steps_per_env_raw = section.get(
        "validation_steps_per_env", defaults.validation_steps_per_env
    )
    validation_steps_per_env = (
        None
        if validation_steps_per_env_raw is None
        else int(validation_steps_per_env_raw)
    )
    if validation_steps_per_env is not None and validation_steps_per_env < 0:
        raise ValueError(
            "`runner.validation_steps_per_env` must be >= 0 or null, "
            f"got {validation_steps_per_env}."
        )

    imitator_rollout_steps_raw = section.get(
        "imitator_rollout_steps_per_env", defaults.imitator_rollout_steps_per_env
    )
    imitator_rollout_steps_per_env = (
        None if imitator_rollout_steps_raw is None else int(imitator_rollout_steps_raw)
    )
    if (
        imitator_rollout_steps_per_env is not None
        and imitator_rollout_steps_per_env <= 0
    ):
        raise ValueError(
            "`runner.imitator_rollout_steps_per_env` must be > 0 or null, "
            f"got {imitator_rollout_steps_per_env}."
        )

    return IrlRunnerCfg(
        steps_per_env_per_cycle=steps_per_env_per_cycle,
        save_interval=save_interval,
        rl_updates_per_cycle=rl_updates_per_cycle,
        reward_updates_per_cycle=reward_updates_per_cycle,
        use_learned_reward=bool(
            section.get("use_learned_reward", defaults.use_learned_reward)
        ),
        use_ground_truth_reward_weights=bool(
            section.get(
                "use_ground_truth_reward_weights",
                defaults.use_ground_truth_reward_weights,
            )
        ),
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
    discount_gamma = (
        float(discount_gamma_raw) if discount_gamma_raw is not None else None
    )
    if discount_gamma is not None and not (0.0 < discount_gamma <= 1.0):
        raise ValueError(
            f"`irl.discount_gamma` must be in (0, 1], got {discount_gamma}."
        )

    expert_num_trajectories_raw = section.get("expert_num_trajectories")
    expert_num_trajectories = (
        int(expert_num_trajectories_raw)
        if expert_num_trajectories_raw is not None
        else None
    )
    if expert_num_trajectories is not None and expert_num_trajectories <= 0:
        raise ValueError(
            f"`irl.expert_num_trajectories` must be > 0 when set, got {expert_num_trajectories}."
        )

    return IRLCfg(
        expert_data_path=str(
            section.get("expert_data_path", defaults.expert_data_path)
        ),
        expert_num_trajectories=expert_num_trajectories,
        expert_subset_strategy=str(
            section.get("expert_subset_strategy", defaults.expert_subset_strategy)
        ),
        batch_size=int(section.get("batch_size", defaults.batch_size)),
        num_learning_epochs=int(
            section.get("num_learning_epochs", defaults.num_learning_epochs)
        ),
        weight_decay=float(section.get("weight_decay", defaults.weight_decay)),
        max_grad_norm=float(section.get("max_grad_norm", defaults.max_grad_norm)),
        reward_loss_coef=float(
            section.get("reward_loss_coef", defaults.reward_loss_coef)
        ),
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
        raise TypeError(
            "`feature_map.ignored_reward_terms` must be a sequence of term names."
        )
    forced_terms = section.get("force_include_terms", [])
    if forced_terms is None:
        forced_terms = []
    if not isinstance(forced_terms, (list, tuple, set)):
        raise TypeError(
            "`feature_map.force_include_terms` must be a sequence of term names."
        )
    return ManagerBasedFeatureCfg(
        ignored_reward_terms=set(str(item) for item in ignored_terms),
        force_include_terms=set(str(item) for item in forced_terms),
    )


def _parse_actor_critic_cfg(section: dict[str, Any]) -> ActorCriticCfg:
    defaults = ActorCriticCfg()
    return ActorCriticCfg(
        actor_hidden_dims=_to_tuple_ints(
            section.get("actor_hidden_dims"), defaults.actor_hidden_dims
        ),
        critic_hidden_dims=_to_tuple_ints(
            section.get("critic_hidden_dims"), defaults.critic_hidden_dims
        ),
        activation=str(section.get("activation", defaults.activation)),
        init_noise_std=float(section.get("init_noise_std", defaults.init_noise_std)),
        noise_std_type=str(section.get("noise_std_type", defaults.noise_std_type)),
        state_dependent_std=bool(
            section.get("state_dependent_std", defaults.state_dependent_std)
        ),
        actor_obs_normalization=bool(
            section.get("actor_obs_normalization", defaults.actor_obs_normalization)
        ),
        critic_obs_normalization=bool(
            section.get("critic_obs_normalization", defaults.critic_obs_normalization)
        ),
    )


def _parse_ppo_cfg(section: dict[str, Any]) -> PpoAlgoCfg:
    defaults = PpoAlgoCfg()
    return PpoAlgoCfg(
        learning_rate=float(section.get("learning_rate", defaults.learning_rate)),
        gamma=float(section.get("gamma", defaults.gamma)),
        lam=float(section.get("lam", defaults.lam)),
        num_learning_epochs=int(
            section.get("num_learning_epochs", defaults.num_learning_epochs)
        ),
        num_mini_batches=int(
            section.get("num_mini_batches", defaults.num_mini_batches)
        ),
        schedule=str(section.get("schedule", defaults.schedule)),
        entropy_coef=float(section.get("entropy_coef", defaults.entropy_coef)),
        desired_kl=float(section.get("desired_kl", defaults.desired_kl)),
        max_grad_norm=float(section.get("max_grad_norm", defaults.max_grad_norm)),
        value_loss_coef=float(section.get("value_loss_coef", defaults.value_loss_coef)),
        use_clipped_value_loss=bool(
            section.get("use_clipped_value_loss", defaults.use_clipped_value_loss)
        ),
        clip_param=float(section.get("clip_param", defaults.clip_param)),
        normalize_advantage_per_mini_batch=bool(
            section.get(
                "normalize_advantage_per_mini_batch",
                defaults.normalize_advantage_per_mini_batch,
            )
        ),
        rnd_cfg=section.get("rnd_cfg", defaults.rnd_cfg),
        symmetry_cfg=section.get("symmetry_cfg", defaults.symmetry_cfg),
    )


def _parse_transition_replay_cfg(
    section: dict[str, Any],
) -> TransitionReplayCfg:
    defaults = TransitionReplayCfg()
    return TransitionReplayCfg(
        learning_starts=int(section.get("learning_starts", defaults.learning_starts)),
        capacity=int(section.get("capacity", defaults.capacity)),
    )


def _parse_off_policy_train_cfg(
    section: dict[str, Any],
) -> OffPolicyTrainCfg:
    defaults = OffPolicyTrainCfg()
    replay = _parse_transition_replay_cfg(
        _to_mapping(section.get("replay"), section_name="algo.train.replay")
    )
    return OffPolicyTrainCfg(
        gamma=float(section.get("gamma", defaults.gamma)),
        rollout_steps=int(section.get("rollout_steps", defaults.rollout_steps)),
        critic_batch_size=int(
            section.get("critic_batch_size", defaults.critic_batch_size)
        ),
        critic_learning_rate=float(
            section.get("critic_learning_rate", defaults.critic_learning_rate)
        ),
        updates_per_transition=float(
            section.get("updates_per_transition", defaults.updates_per_transition)
        ),
        initial_critic_updates=int(
            section.get("initial_critic_updates", defaults.initial_critic_updates)
        ),
        target_polyak=float(section.get("target_polyak", defaults.target_polyak)),
        max_grad_norm=float(section.get("max_grad_norm", defaults.max_grad_norm)),
        replay=replay,
    )


def _parse_tanh_policy_cfg(
    section: dict[str, Any],
) -> TanhGaussianPolicyCfg:
    defaults = TanhGaussianPolicyCfg()
    return TanhGaussianPolicyCfg(
        hidden_dims=_to_tuple_ints(section.get("hidden_dims"), defaults.hidden_dims),
        initial_std=float(section.get("initial_std", defaults.initial_std)),
        minimum_std=float(section.get("minimum_std", defaults.minimum_std)),
        maximum_std=float(section.get("maximum_std", defaults.maximum_std)),
        action_bound=float(section.get("action_bound", defaults.action_bound)),
    )


def _parse_value_cfg(
    section: dict[str, Any],
) -> DirectQCfg | SuccessorQCfg:
    value_type = str(section.get("type", "direct"))
    if value_type == "direct":
        defaults = DirectQCfg()
        return DirectQCfg(
            num_heads=int(section.get("num_heads", defaults.num_heads)),
            hidden_dims=_to_tuple_ints(
                section.get("hidden_dims"), defaults.hidden_dims
            ),
        )
    if value_type == "successor":
        defaults = SuccessorQCfg()
        return SuccessorQCfg(
            num_heads=int(section.get("num_heads", defaults.num_heads)),
            trunk_hidden_dims=_to_tuple_ints(
                section.get("trunk_hidden_dims"), defaults.trunk_hidden_dims
            ),
            latent_dim=int(section.get("latent_dim", defaults.latent_dim)),
            head_hidden_dim=int(
                section.get("head_hidden_dim", defaults.head_hidden_dim)
            ),
            return_scale_min=float(
                section.get("return_scale_min", defaults.return_scale_min)
            ),
            entropy_critic_hidden_dims=_to_tuple_ints(
                section.get("entropy_critic_hidden_dims"),
                defaults.entropy_critic_hidden_dims,
            ),
            entropy_critic_learning_rate=float(
                section.get(
                    "entropy_critic_learning_rate",
                    defaults.entropy_critic_learning_rate,
                )
            ),
        )
    raise ValueError(
        f"algo.value.type must be 'direct' or 'successor', got {value_type!r}."
    )


def _parse_algorithm_cfg(
    section: dict[str, Any],
) -> PpoAlgoCfg | SacCfg | PmdCfg:
    algorithm_type = str(section.get("type", "ppo"))
    if algorithm_type == "ppo":
        return _parse_ppo_cfg(section)
    train = _parse_off_policy_train_cfg(
        _to_mapping(section.get("train"), section_name="algo.train")
    )
    policy = _parse_tanh_policy_cfg(
        _to_mapping(section.get("policy"), section_name="algo.policy")
    )
    value = _parse_value_cfg(
        _to_mapping(section.get("value"), section_name="algo.value")
    )
    if algorithm_type == "sac":
        defaults = SacCfg()
        return SacCfg(
            train=train,
            policy=policy,
            value=value,
            actor_learning_rate=float(
                section.get("actor_learning_rate", defaults.actor_learning_rate)
            ),
            actor_batch_size=int(
                section.get("actor_batch_size", defaults.actor_batch_size)
            ),
            initial_entropy_temperature=float(
                section.get(
                    "initial_entropy_temperature",
                    defaults.initial_entropy_temperature,
                )
            ),
            automatic_entropy_tuning=bool(
                section.get(
                    "automatic_entropy_tuning",
                    defaults.automatic_entropy_tuning,
                )
            ),
            entropy_learning_rate=float(
                section.get("entropy_learning_rate", defaults.entropy_learning_rate)
            ),
        )
    if algorithm_type == "pmd":
        defaults = PmdCfg()
        return PmdCfg(
            train=train,
            policy=policy,
            value=value,
            entropy_temperature=float(
                section.get("entropy_temperature", defaults.entropy_temperature)
            ),
            beta=float(section.get("beta", defaults.beta)),
            candidate_actions=int(
                section.get("candidate_actions", defaults.candidate_actions)
            ),
            candidate_target_kl=float(
                section.get("candidate_target_kl", defaults.candidate_target_kl)
            ),
            actor_projection_kl=float(
                section.get("actor_projection_kl", defaults.actor_projection_kl)
            ),
            q_coefficient_min=float(
                section.get("q_coefficient_min", defaults.q_coefficient_min)
            ),
            q_coefficient_max=float(
                section.get("q_coefficient_max", defaults.q_coefficient_max)
            ),
            coefficient_bisection_steps=int(
                section.get(
                    "coefficient_bisection_steps",
                    defaults.coefficient_bisection_steps,
                )
            ),
            actor_learning_rate=float(
                section.get("actor_learning_rate", defaults.actor_learning_rate)
            ),
            actor_batch_size=int(
                section.get("actor_batch_size", defaults.actor_batch_size)
            ),
            projection_steps=int(
                section.get("projection_steps", defaults.projection_steps)
            ),
            projection_backtrack_steps=int(
                section.get(
                    "projection_backtrack_steps", defaults.projection_backtrack_steps
                )
            ),
            critic_updates_per_projection=int(
                section.get(
                    "critic_updates_per_projection",
                    defaults.critic_updates_per_projection,
                )
            ),
        )
    raise ValueError(f"algo.type must be one of ppo, sac, pmd; got {algorithm_type!r}.")


def _parse_bc_cfg(section: dict[str, Any]) -> BCCfg:
    defaults = BCCfg()
    weight = float(section.get("weight", defaults.weight))
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"`bc.weight` must be in [0, 1], got {weight}.")
    loss_type = str(section.get("loss_type", defaults.loss_type))
    if loss_type not in {"nll", "mse"}:
        raise ValueError(f"`bc.loss_type` must be 'nll' or 'mse', got {loss_type!r}.")
    batch_size = int(section.get("batch_size", defaults.batch_size))
    if batch_size <= 0:
        raise ValueError(f"`bc.batch_size` must be > 0, got {batch_size}.")
    val_fraction = float(section.get("val_fraction", defaults.val_fraction))
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"`bc.val_fraction` must be in [0, 1), got {val_fraction}.")
    return BCCfg(
        weight=weight,
        loss_type=loss_type,
        batch_size=batch_size,
        val_fraction=val_fraction,
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
    linear_projection = str(
        section.get("linear_projection", defaults.linear_projection)
    )
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
        regularization_strength=float(
            section.get("regularization_strength", defaults.regularization_strength)
        ),
        elastic_alpha=float(section.get("elastic_alpha", defaults.elastic_alpha)),
        linear_projection=linear_projection,
        linear_projection_radius=float(
            section.get("linear_projection_radius", defaults.linear_projection_radius)
        ),
    )


def load_train_cfg(
    *,
    task_name: str,
    args_cli: argparse.Namespace,
    config_dir: Path,
) -> TrainCfg:
    """Load and validate a :class:`TrainCfg` from YAML + CLI overrides.

    ``config_dir`` must point at the directory containing ``experiment.yaml``.
    Callers (e.g. ``scripts/irl/train_irl.py``) compute the default location
    and pass it in explicitly — there is no implicit repo-root lookup here.
    """
    payload = _load_train_payload(task_name=task_name, config_dir=config_dir)

    env_section = _to_mapping(payload.get("env"), section_name="env")
    resolved_task_name = str(task_name or env_section.get("name") or "")
    if resolved_task_name == "":
        raise ValueError(
            "Task name is required. Provide `--task` or set `env.name` in train config."
        )

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
            num_envs=(
                int(env_section["num_envs"])
                if env_section.get("num_envs") is not None
                else None
            ),
            success_threshold=float(
                env_section.get("success_threshold", env_defaults.success_threshold)
            ),
            add_success_bonus=bool(
                env_section.get("add_success_bonus", env_defaults.add_success_bonus)
            ),
            success_bonus_weight=float(
                env_section.get(
                    "success_bonus_weight", env_defaults.success_bonus_weight
                )
            ),
        ),
    )

    cfg.feature_map = _parse_feature_map_cfg(
        _to_mapping(payload.get("feature_map"), section_name="feature_map")
    )
    cfg.irl = _parse_irl_cfg(_to_mapping(payload.get("irl"), section_name="irl"))
    cfg.runner = _parse_runner_cfg(
        _to_mapping(payload.get("runner"), section_name="runner")
    )
    cfg.reward = _parse_reward_cfg(
        _to_mapping(payload.get("reward"), section_name="reward")
    )
    cfg.actor_critic = _parse_actor_critic_cfg(
        _to_mapping(payload.get("policy"), section_name="policy")
    )
    algorithm_section = _to_mapping(payload.get("algo"), section_name="algo")
    cli_algorithm = getattr(args_cli, "algorithm", None)
    if cli_algorithm is not None:
        algorithm_section = dict(algorithm_section)
        algorithm_section["type"] = str(cli_algorithm)
    cfg.rl_algorithm = _parse_algorithm_cfg(algorithm_section)
    cfg.bc = _parse_bc_cfg(_to_mapping(payload.get("bc"), section_name="bc"))

    _apply_cli_overrides(cfg, args_cli)
    if cfg.resume and cfg.init_policy_checkpoint is not None:
        raise ValueError(
            "--init_policy_checkpoint is mutually exclusive with --resume."
        )
    return cfg


def _apply_cli_overrides(cfg: TrainCfg, args_cli: argparse.Namespace) -> None:
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
    algorithm = getattr(args_cli, "algorithm", None)
    if algorithm is not None:
        if algorithm not in {"ppo", "sac", "pmd"}:
            raise ValueError(f"Unsupported algorithm override {algorithm!r}.")
        if algorithm != "ppo":
            cfg.runner = replace(
                cfg.runner, reward_updates_per_cycle=8, rl_updates_per_cycle=1
            )

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
    init_checkpoint = getattr(args_cli, "init_policy_checkpoint", None)
    if init_checkpoint is not None:
        cfg.init_policy_checkpoint = str(init_checkpoint)
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

    use_ground_truth_reward_weights = getattr(
        args_cli, "use_ground_truth_reward_weights", None
    )
    if use_ground_truth_reward_weights is not None:
        cfg.runner = replace(
            cfg.runner,
            use_ground_truth_reward_weights=bool(use_ground_truth_reward_weights),
        )

    bc_weight = getattr(args_cli, "bc_weight", None)
    if bc_weight is not None:
        bc_weight = float(bc_weight)
        if not 0.0 <= bc_weight <= 1.0:
            raise ValueError(f"`--bc_weight` must be in [0, 1], got {bc_weight}.")
        cfg.bc = replace(cfg.bc, weight=bc_weight)
    bc_loss_type = getattr(args_cli, "bc_loss_type", None)
    if bc_loss_type is not None:
        cfg.bc = replace(cfg.bc, loss_type=str(bc_loss_type))

    learning_rate = getattr(args_cli, "learning_rate", None)
    if learning_rate is not None:
        learning_rate = float(learning_rate)
        if learning_rate <= 0.0:
            raise ValueError(f"`--learning_rate` must be > 0, got {learning_rate}.")
        if isinstance(cfg.rl_algorithm, PpoAlgoCfg):
            cfg.rl_algorithm = replace(cfg.rl_algorithm, learning_rate=learning_rate)
        else:
            cfg.rl_algorithm = replace(
                cfg.rl_algorithm, actor_learning_rate=learning_rate
            )

    off_policy_override_names = (
        "value_backend",
        "num_value_heads",
        "updates_per_transition",
        "initial_critic_updates",
        "critic_learning_rate",
        "rollout_steps",
        "pmd_beta",
        "pmd_critic_updates_per_projection",
        "pmd_candidate_target_kl",
        "pmd_actor_projection_kl",
        "entropy_temperature",
    )
    requested_off_policy_overrides = {
        name: getattr(args_cli, name, None) for name in off_policy_override_names
    }
    if any(value is not None for value in requested_off_policy_overrides.values()):
        if not isinstance(cfg.rl_algorithm, (SacCfg, PmdCfg)):
            raise ValueError(
                "Off-policy CLI overrides require --algorithm sac or --algorithm pmd."
            )

        value_cfg = cfg.rl_algorithm.value
        value_backend = requested_off_policy_overrides["value_backend"]
        num_value_heads = requested_off_policy_overrides["num_value_heads"]
        if value_backend is not None and value_backend != value_cfg.type:
            value_cfg = DirectQCfg() if value_backend == "direct" else SuccessorQCfg()
        if num_value_heads is not None:
            if int(num_value_heads) <= 0:
                raise ValueError("--num_value_heads must be > 0.")
            value_cfg = replace(value_cfg, num_heads=int(num_value_heads))

        train_overrides: dict[str, float | int] = {}
        updates_per_transition = requested_off_policy_overrides[
            "updates_per_transition"
        ]
        if updates_per_transition is not None:
            train_overrides["updates_per_transition"] = float(updates_per_transition)
        initial_critic_updates = requested_off_policy_overrides[
            "initial_critic_updates"
        ]
        if initial_critic_updates is not None:
            train_overrides["initial_critic_updates"] = int(initial_critic_updates)
        critic_learning_rate = requested_off_policy_overrides["critic_learning_rate"]
        if critic_learning_rate is not None:
            train_overrides["critic_learning_rate"] = float(critic_learning_rate)
        rollout_steps = requested_off_policy_overrides["rollout_steps"]
        if rollout_steps is not None:
            train_overrides["rollout_steps"] = int(rollout_steps)
        cfg.rl_algorithm = replace(
            cfg.rl_algorithm,
            value=value_cfg,
            train=replace(cfg.rl_algorithm.train, **train_overrides),
        )

        pmd_overrides: dict[str, float | int] = {}
        pmd_beta = requested_off_policy_overrides["pmd_beta"]
        if pmd_beta is not None:
            pmd_overrides["beta"] = float(pmd_beta)
        critic_updates_per_projection = requested_off_policy_overrides[
            "pmd_critic_updates_per_projection"
        ]
        if critic_updates_per_projection is not None:
            pmd_overrides["critic_updates_per_projection"] = int(
                critic_updates_per_projection
            )
        candidate_target_kl = requested_off_policy_overrides["pmd_candidate_target_kl"]
        if candidate_target_kl is not None:
            pmd_overrides["candidate_target_kl"] = float(candidate_target_kl)
        actor_projection_kl = requested_off_policy_overrides["pmd_actor_projection_kl"]
        if actor_projection_kl is not None:
            pmd_overrides["actor_projection_kl"] = float(actor_projection_kl)
        if pmd_overrides:
            if not isinstance(cfg.rl_algorithm, PmdCfg):
                raise ValueError(
                    "PMD overrides ("
                    + ", ".join(sorted(pmd_overrides))
                    + ") require --algorithm pmd."
                )
            cfg.rl_algorithm = replace(cfg.rl_algorithm, **pmd_overrides)

        entropy_temperature = requested_off_policy_overrides["entropy_temperature"]
        if entropy_temperature is not None:
            temperature = float(entropy_temperature)
            if isinstance(cfg.rl_algorithm, SacCfg):
                cfg.rl_algorithm = replace(
                    cfg.rl_algorithm,
                    initial_entropy_temperature=temperature,
                    automatic_entropy_tuning=(
                        cfg.rl_algorithm.automatic_entropy_tuning and temperature > 0.0
                    ),
                )
            else:
                cfg.rl_algorithm = replace(
                    cfg.rl_algorithm, entropy_temperature=temperature
                )

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
