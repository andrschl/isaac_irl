# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train IRL agent with PPO + feature-buffer IRL.

This file is the entry point. The heavy lifting is split across sibling
modules in this directory:

- ``_train_config.py``  — config dataclasses + YAML/CLI parsers + run-config dumping
- ``_train_expert_data.py`` — HDF5/.pt expert loaders + subset/holdout helpers + ``_install_expert_episodes``
- ``_train_video.py``   — ``VideoSetup`` + ``setup_video_recording`` + ``copy_final_videos`` + ``rename_iteration_files``

This file owns:
- ``RuntimeDeps`` (the Isaac/rsl_rl handle, populated only after ``SimulationApp`` exists)
- ``_load_app_launcher_cls`` / ``_load_runtime_deps`` — lazy Isaac/rsl_rl loading
- Argparse setup
- Wiring builders (actor/critic, feature map, success_fn, BC, PPO+adapter, wandb cfg)
- ``main()`` — orchestrates setup, training, cleanup
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

# Load .env for WANDB_API_KEY before any wandb imports
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

import torch

# Make sibling files importable both when this script is run as
# ``python scripts/irl/train_irl.py`` and when tests load it via importlib.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from algorithms import FeatureRewardLearner, IRLCfg
from algorithms.bc import BC, BCCfg
from algorithms.ppo_with_bc import make_ppo_with_bc_cls
from reward_features.manager_based import (
    ManagerBasedFeatureCfg,
    manager_based_reward_feature_dict,
    manager_based_reward_features,
)
from reward_features.success_bonus import add_success_bonus_term
from reward_model import DenseFeatureRewardModel, LinearFeatureRewardModel, RewardModelCfg
from runner import IrlRunner, IrlRunnerCfg, NoopMetricLogger, RslRlPpoAdapter, WandbMetricLogger
from utils.runtime_context import RuntimeContext
from storage.feature_storage import FeatureBufCfg
from storage.obs_action_storage import ObsActionBufCfg, ObsActionBuffer

from _train_config import (
    ActorCriticCfg,
    EnvCfg,
    PpoAlgoCfg,
    TrainCfg,
    _dump_run_configs,
    load_train_cfg,
)
from _train_expert_data import (
    _install_expert_episodes,
    _split_holdout_episodes,
    _subset_episodes,
    load_expert_success_rate,
    load_feature_episodes,
    load_obs_action_episodes,
)
from _train_video import (
    VideoSetup,
    copy_final_videos,
    rename_iteration_files,
    setup_video_recording,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Runtime deps (must be populated AFTER SimulationApp creation)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RuntimeDeps:
    AppLauncher: Any
    gym: Any
    parse_env_cfg: Callable[..., Any]
    get_checkpoint_path: Callable[..., str]
    RslRlVecEnvWrapper: Any
    PPO: Any
    MLPModel: Any
    RolloutStorage: Any
    resolve_obs_groups: Callable[..., Any]


def _load_app_launcher_cls() -> Any:
    from isaaclab.app import AppLauncher

    return AppLauncher


def _load_runtime_deps(app_launcher_cls: Any) -> RuntimeDeps:
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from rsl_rl.utils import resolve_obs_groups

    return RuntimeDeps(
        AppLauncher=app_launcher_cls,
        gym=gym,
        parse_env_cfg=parse_env_cfg,
        get_checkpoint_path=get_checkpoint_path,
        RslRlVecEnvWrapper=RslRlVecEnvWrapper,
        PPO=PPO,
        MLPModel=MLPModel,
        RolloutStorage=RolloutStorage,
        resolve_obs_groups=resolve_obs_groups,
    )


# ---------------------------------------------------------------------------
# Wiring builders (actor/critic, feature map, success_fn, log dir, BC, RL alg, wandb cfg)
# ---------------------------------------------------------------------------


def _build_actor_critic(
    *,
    obs: Any,
    obs_groups: dict[str, list[str]],
    num_actions: int,
    actor_critic_cfg: ActorCriticCfg,
    device: str,
    MLPModel: Any,
) -> tuple[Any, Any]:
    """Construct rsl_rl 5.x actor + critic MLPModels from the legacy ActorCriticCfg.

    Translates the (init_noise_std, noise_std_type, state_dependent_std) trio
    that the YAML still uses into the new ``distribution_cfg`` dict consumed by
    MLPModel. Only Gaussian (non-state-dependent) is supported here — matches
    what the franka_lift config provides.
    """
    if actor_critic_cfg.state_dependent_std:
        raise NotImplementedError(
            "state_dependent_std=True is not wired into the new MLPModel construction yet. "
            "Add a HeteroscedasticGaussianDistribution branch if you need it."
        )
    distribution_cfg = {
        "class_name": "rsl_rl.modules.GaussianDistribution",
        "init_std": float(actor_critic_cfg.init_noise_std),
        "std_type": str(actor_critic_cfg.noise_std_type),
    }
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        int(num_actions),
        hidden_dims=list(actor_critic_cfg.actor_hidden_dims),
        activation=str(actor_critic_cfg.activation),
        obs_normalization=bool(actor_critic_cfg.actor_obs_normalization),
        distribution_cfg=distribution_cfg,
    ).to(device)
    critic = MLPModel(
        obs,
        obs_groups,
        "critic",
        1,
        hidden_dims=list(actor_critic_cfg.critic_hidden_dims),
        activation=str(actor_critic_cfg.activation),
        obs_normalization=bool(actor_critic_cfg.critic_obs_normalization),
        distribution_cfg=None,
    ).to(device)
    return actor, critic


def _build_feature_map(
    cfg: ManagerBasedFeatureCfg,
    device: str | torch.device,
) -> Callable[[Any], torch.Tensor]:
    def _feature_map(env: Any) -> torch.Tensor:
        return manager_based_reward_features(
            env=env,
            ignored_reward_terms=cfg.ignored_reward_terms,
            device=device,
            force_include_terms=cfg.force_include_terms,
        )

    return _feature_map


def _build_success_fn(
    *,
    task_name: str,
    threshold: float,
    probe_env: Any,
) -> Callable[[Any], torch.Tensor] | None:
    """Build a ``success_fn(env) -> [N] bool`` for the validation success metric.

    Uses the Lift task's ``object_reached_goal`` (object within ``threshold`` of
    the commanded goal pose). Returns None — disabling success logging — for any
    task that doesn't expose it, or if a probe call fails. Best-effort: success
    logging must never break training.
    """
    try:
        from isaaclab_tasks.manager_based.manipulation.lift.mdp import (
            object_reached_goal,
        )
    except Exception as exc:
        print(
            f"[WARN] No success metric for task {task_name!r} (object_reached_goal unavailable): {exc}"
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


def _feature_dim_from_feature_map(
    feature_map: Callable[[Any], torch.Tensor], env: Any
) -> int:
    with torch.no_grad():
        features = feature_map(env)
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 2:
        raise ValueError(
            f"Feature map must return shape [N, D], got {tuple(features.shape)}"
        )
    return int(features.shape[1])


def _build_log_dir(experiment_name: str, run_name: str) -> tuple[str, str]:
    from datetime import datetime

    log_root_path = os.path.abspath(os.path.join("logs", "irl", experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        log_dir += f"_{run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return log_root_path, log_dir


def _get_initial_obs(
    env: Any,
    resolve_obs_groups_fn: Callable[..., dict[str, list[str]]],
) -> tuple[Any, dict[str, list[str]]]:
    obs = env.get_observations()
    if not hasattr(obs, "keys"):
        raise TypeError(
            "RslRlVecEnvWrapper.get_observations() must return TensorDict-like observations."
        )
    obs_groups = resolve_obs_groups_fn(obs, {}, default_sets=["actor", "critic"])
    return obs, obs_groups


def _load_expert_success_rate_with_logging(expert_data_path: str) -> float | None:
    """Read the expert success rate from demos, logging the outcome."""
    try:
        rate = load_expert_success_rate(os.path.abspath(expert_data_path))
    except Exception as exc:
        print(f"[WARN] Could not read expert success rate from demos: {exc}")
        return None
    if rate is None:
        print("[INFO] No expert success info in demos; expert-success gap disabled.")
    else:
        print(f"[INFO] Expert success rate (from demos): {rate:.4f}")
    return rate


def _build_bc_alg(
    train_cfg: TrainCfg,
    *,
    runtime_ctx: RuntimeContext,
) -> BC | None:
    """Construct the BC loss provider from expert (obs, action) demos.

    Returns ``None`` when BC is disabled (``bc.alpha == 0``). Otherwise loads
    demos, applies the optional trajectory subset, splits a validation hold-out,
    and wraps the train/val buffers in a :class:`BC` instance.
    """
    if train_cfg.bc.alpha <= 0.0:
        return None

    obs_action_episodes = load_obs_action_episodes(train_cfg.irl.expert_data_path)
    if train_cfg.irl.expert_num_trajectories is not None:
        total = len(obs_action_episodes)
        obs_action_episodes = _subset_episodes(
            obs_action_episodes,
            int(train_cfg.irl.expert_num_trajectories),
            train_cfg.irl.expert_subset_strategy,
            train_cfg.seed,
        )
        print(
            f"[INFO] BC expert subset: {len(obs_action_episodes)}/{total} "
            f"trajectories (strategy={train_cfg.irl.expert_subset_strategy})"
        )

    train_episodes, val_episodes = _split_holdout_episodes(
        obs_action_episodes, train_cfg.bc.val_fraction, train_cfg.seed
    )
    buf_cfg = ObsActionBufCfg(store_device="cpu", store_dtype=torch.float32)
    train_storage = ObsActionBuffer(cfg=buf_cfg, ctx=runtime_ctx)
    train_storage.load_episodes(train_episodes)

    val_storage: ObsActionBuffer | None = None
    if val_episodes:
        val_storage = ObsActionBuffer(cfg=buf_cfg, ctx=runtime_ctx)
        val_storage.load_episodes(val_episodes)

    val_suffix = (
        f", {len(val_episodes)} val episodes ({len(val_storage)} pairs)"
        if val_storage is not None
        else " (no validation split)"
    )
    print(
        f"[INFO] BC dataset: {len(train_episodes)} train episodes "
        f"({len(train_storage)} pairs){val_suffix}."
    )
    return BC(
        cfg=train_cfg.bc,
        storage=train_storage,
        val_storage=val_storage,
        device=train_cfg.device,
    )


def _build_rl_alg(
    *,
    actor: Any,
    critic: Any,
    obs: Any,
    env: Any,
    train_cfg: TrainCfg,
    bc_alg: BC | None,
    runtime_deps: RuntimeDeps,
) -> RslRlPpoAdapter:
    """Build the rsl_rl PPO (optionally with BC) and wrap it in the adapter."""
    rollout_storage = runtime_deps.RolloutStorage(
        "rl",
        int(env.num_envs),
        int(train_cfg.runner.steps_per_env_per_cycle),
        obs,
        [int(env.num_actions)],
        train_cfg.device,
    )
    ppo_kwargs = asdict(train_cfg.rl_algorithm)
    if bc_alg is None:
        ppo_alg = runtime_deps.PPO(
            actor=actor,
            critic=critic,
            storage=rollout_storage,
            device=train_cfg.device,
            **ppo_kwargs,
        )
    else:
        ppo_cls = make_ppo_with_bc_cls(runtime_deps.PPO)
        ppo_alg = ppo_cls(
            actor=actor,
            critic=critic,
            storage=rollout_storage,
            device=train_cfg.device,
            bc_alg=bc_alg,
            **ppo_kwargs,
        )
    return RslRlPpoAdapter(ppo_alg, bc_alg=bc_alg)


def _build_wandb_config(
    train_cfg: TrainCfg, *, num_envs: int
) -> tuple[dict[str, Any], list[str]]:
    """Assemble the wandb run config + tags from a parsed TrainCfg."""
    config = {
        "task": str(train_cfg.env.name),
        "num_envs": int(num_envs),
        "seed": int(train_cfg.seed),
        "max_iterations": int(train_cfg.max_iterations),
        "bc/alpha": float(train_cfg.bc.alpha),
        "bc/loss_type": str(train_cfg.bc.loss_type),
        "bc/batch_size": int(train_cfg.bc.batch_size),
        "bc/val_fraction": float(train_cfg.bc.val_fraction),
        "irl/expert_data_path": str(train_cfg.irl.expert_data_path),
        "irl/expert_num_trajectories": train_cfg.irl.expert_num_trajectories,
        "irl/discount_gamma": train_cfg.irl.discount_gamma,
        "irl/reward_learning_rate": train_cfg.irl.reward_learning_rate,
        "irl/batch_size": int(train_cfg.irl.batch_size),
        "irl/num_learning_epochs": int(train_cfg.irl.num_learning_epochs),
        "runner/use_learned_reward": bool(train_cfg.runner.use_learned_reward),
        "runner/reward_updates_per_cycle": int(
            train_cfg.runner.reward_updates_per_cycle
        ),
        "runner/rl_updates_per_cycle": int(train_cfg.runner.rl_updates_per_cycle),
        "ppo/learning_rate": float(train_cfg.rl_algorithm.learning_rate),
        "ppo/gamma": float(train_cfg.rl_algorithm.gamma),
        "ppo/lam": float(train_cfg.rl_algorithm.lam),
        "reward/type": "linear" if train_cfg.reward.is_linear else "dense",
        "reward/regularization": str(train_cfg.reward.regularization),
        "reward/regularization_strength": float(
            train_cfg.reward.regularization_strength
        ),
    }
    tags = [
        f"alpha={train_cfg.bc.alpha}",
        f"seed={train_cfg.seed}",
        f"loss={train_cfg.bc.loss_type}",
    ]
    return config, tags


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_arg_parser(app_launcher_cls: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an IRL agent with PPO + feature-buffer IRL."
    )

    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record videos during training.",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=1500,
        help="Length of recorded videos (steps).",
    )
    parser.add_argument(
        "--video_interval",
        type=int,
        default=2000,
        help="Interval between recordings (steps).",
    )
    parser.add_argument(
        "--video_interval_iterations",
        type=int,
        default=None,
        help="Interval between recordings (learning iterations). Overrides --video_interval.",
    )
    parser.add_argument(
        "--video_final_only",
        action="store_true",
        default=False,
        help="Record only one clip near the end of training (overrides --video_interval*).",
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    parser.add_argument("--task", type=str, default=None, help="Task name.")
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Environment / training seed."
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None, help="Training iterations."
    )
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
    parser.add_argument(
        "--expert_num_trajectories",
        type=int,
        default=None,
        help="Max expert trajectories to use (subset for ablations). None = use all.",
    )
    parser.add_argument(
        "--expert_subset_strategy",
        type=str,
        choices={"first", "random"},
        default=None,
        help="Subset strategy when expert_num_trajectories is set: first | random.",
    )
    parser.add_argument(
        "--bc_alpha",
        type=float,
        default=None,
        help="Convex mix on the policy loss: (1-α)·L_PPO + α·L_BC. 0 = pure PPO+IRL, 1 = pure BC.",
    )
    parser.add_argument(
        "--bc_loss_type",
        type=str,
        choices={"nll", "mse"},
        default=None,
        help="BC loss type: nll (default) or mse.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override the actor/PPO optimizer learning rate. BC shares PPO's optimizer, "
        "so this is also the BC learning rate.",
    )
    parser.add_argument(
        "--reward_learning_rate",
        type=float,
        default=None,
        help="Override `irl.reward_learning_rate` (AdamW LR for the reward model).",
    )
    parser.add_argument(
        "--reward_regularization_strength",
        type=float,
        default=None,
        help="Override `reward.regularization_strength` (coef on the reward-weight regularizer).",
    )
    parser.add_argument(
        "--reward_updates_per_cycle",
        type=int,
        default=None,
        help="Override `runner.reward_updates_per_cycle` (reward-grad steps per PPO iter).",
    )
    parser.add_argument(
        "--rl_updates_per_cycle",
        type=int,
        default=None,
        help="Override `runner.rl_updates_per_cycle` (agent update calls per iteration).",
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=None,
        help="Override `env.success_threshold` from the yaml config. "
        "Object-to-goal distance (m) below which a lift episode counts as a success.",
    )
    parser.add_argument(
        "--validation_interval",
        type=int,
        default=None,
        help="Override `runner.validation_interval` (env-rollout validation cadence, "
        "in training iterations). Bump for BC-only sweeps where each iter is cheap "
        "but the validation rollout dominates wall-clock. 0 disables validation.",
    )
    parser.add_argument(
        "--use_learned_reward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override `runner.use_learned_reward`. When True (default in yaml), the agent is "
        "trained against the IRL reward model — actual IRL. When False, it trains "
        "against the env's GT reward and the IRL reward model is fit purely as a "
        "diagnostic.",
    )
    parser.add_argument(
        "--add_success_bonus",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override `env.add_success_bonus`. When set, a `success_bonus` term "
        "(binary object_reached_goal) is injected into the env's RewardsCfg and "
        "auto-registered as an IRL feature. Env-reward contribution is controlled "
        "by --success_bonus_weight (default 0.0 → feature-only).",
    )
    parser.add_argument(
        "--success_bonus_weight",
        type=float,
        default=None,
        help="Override `env.success_bonus_weight` — env-reward weight of the injected "
        "success-bonus term. 0 = feature-only (recommended); >0 also adds to env reward.",
    )

    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Experiment name for logging."
    )
    parser.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume from a previous checkpoint.",
    )
    parser.add_argument(
        "--load_run", type=str, default=None, help="Run folder to resume from."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint file to resume from."
    )
    parser.add_argument(
        "--logger",
        type=str,
        choices={"wandb", "noop"},
        default=None,
        help="Logger backend.",
    )
    parser.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Project name for wandb logger.",
    )

    app_launcher_cls.add_app_launcher_args(parser)
    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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
    runner: IrlRunner | None = None
    log_dir: str | None = None
    video = VideoSetup()
    run_crashed = False

    if runtime_deps is None:
        # Isaac/Omniverse runtime deps must be imported only after SimulationApp creation.
        runtime_deps = _load_runtime_deps(app_launcher_cls=app_launcher_cls)

    try:
        requested_task_name = args_cli.task
        if requested_task_name is None:
            raise ValueError("`--task` is required.")

        train_cfg = load_train_cfg(task_name=requested_task_name, args_cli=args_cli)

        if not train_cfg.irl.expert_data_path:
            raise ValueError(
                "`irl.expert_data_path` is required for train_irl.py. "
                "Pass --expert_data_path or set it in configs/franka_lift/experiment.yaml."
            )

        env_cfg = runtime_deps.parse_env_cfg(
            requested_task_name,
            device=train_cfg.env.device,
            num_envs=train_cfg.env.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )

        if train_cfg.env.add_success_bonus:
            added = add_success_bonus_term(
                env_cfg,
                threshold=float(train_cfg.env.success_threshold),
                weight=float(train_cfg.env.success_bonus_weight),
            )
            if added:
                # Ensure the feature extractor picks it up even at weight=0
                # (feature-only mode).
                train_cfg.feature_map.force_include_terms.add("success_bonus")
                print(
                    f"[INFO] Injected `success_bonus` reward term "
                    f"(env_weight={train_cfg.env.success_bonus_weight}, "
                    f"threshold={train_cfg.env.success_threshold} m, "
                    f"feature-only={train_cfg.env.success_bonus_weight == 0.0})."
                )
            else:
                print(
                    "[WARN] add_success_bonus=True but env_cfg has no `rewards` section; skipped."
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

        env, video = setup_video_recording(
            env,
            args_cli=args_cli,
            log_dir=log_dir,
            max_iterations=int(train_cfg.max_iterations),
            steps_per_env_per_cycle=int(train_cfg.runner.steps_per_env_per_cycle),
            gym_wrappers=runtime_deps.gym.wrappers,
        )

        env = runtime_deps.RslRlVecEnvWrapper(env)
        if hasattr(env, "seed"):
            env.seed(int(train_cfg.seed))

        obs, obs_groups = _get_initial_obs(env, runtime_deps.resolve_obs_groups)
        obs = obs.to(train_cfg.device)
        actor, critic = _build_actor_critic(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=int(env.num_actions),
            actor_critic_cfg=train_cfg.actor_critic,
            device=train_cfg.device,
            MLPModel=runtime_deps.MLPModel,
        )

        feature_map = _build_feature_map(train_cfg.feature_map, device=train_cfg.device)
        feature_dim = _feature_dim_from_feature_map(feature_map, env)
        runtime_ctx = RuntimeContext(
            num_envs=int(env.num_envs),
            feature_dim=feature_dim,
            device=train_cfg.device,
        )
        feature_names = list(
            manager_based_reward_feature_dict(
                env,
                ignored_reward_terms=train_cfg.feature_map.ignored_reward_terms,
                device=train_cfg.device,
                force_include_terms=train_cfg.feature_map.force_include_terms,
            ).keys()
        )

        # Optional task success predicate for the validation success-rate metric.
        # Uses the Lift task's built-in `object_reached_goal` (object within
        # `env.success_threshold` of the commanded goal pose). Disabled gracefully
        # for tasks that don't provide it.
        success_fn = _build_success_fn(
            task_name=requested_task_name,
            threshold=float(train_cfg.env.success_threshold),
            probe_env=env,
        )

        bc_alg = _build_bc_alg(train_cfg, runtime_ctx=runtime_ctx)
        rl_alg = _build_rl_alg(
            actor=actor,
            critic=critic,
            obs=obs,
            env=env,
            train_cfg=train_cfg,
            bc_alg=bc_alg,
            runtime_deps=runtime_deps,
        )
        reward_cfg = replace(train_cfg.reward, num_features=runtime_ctx.feature_dim)
        reward_model_cls = (
            LinearFeatureRewardModel
            if reward_cfg.is_linear
            else DenseFeatureRewardModel
        )
        reward_model = reward_model_cls(reward_cfg).to(train_cfg.device)
        expert_success_rate = _load_expert_success_rate_with_logging(
            train_cfg.irl.expert_data_path
        )
        wandb_config, wandb_tags = _build_wandb_config(
            train_cfg, num_envs=int(env.num_envs)
        )

        if train_cfg.runner.use_learned_reward:
            print("[INFO] IRL mode: agent trains against the learned reward model.")
        else:
            print(
                "[INFO] IRL mode: agent trains against the env's GT reward "
                "(learned reward model fit as diagnostic only)."
            )

        irl_alg = FeatureRewardLearner(
            reward=reward_model,
            gamma=float(train_cfg.rl_algorithm.gamma),
            cfg=train_cfg.irl,
            device=train_cfg.device,
            feature_names=feature_names,
            expert_success_rate=expert_success_rate,
        )
        metric_logger = (
            WandbMetricLogger(
                project=train_cfg.wandb_project,
                log_dir=log_dir,
                run_name=(train_cfg.run_name or None),
                group=train_cfg.experiment_name,
                tags=wandb_tags,
                config=wandb_config,
            )
            if train_cfg.logger == "wandb"
            else NoopMetricLogger()
        )

        runner = IrlRunner(
            env=env,
            rl_alg=rl_alg,
            irl_alg=irl_alg,
            feature_map=feature_map,
            runner_cfg=train_cfg.runner,
            success_fn=success_fn,
            runtime_ctx=runtime_ctx,
            log_dir=log_dir,
            device=train_cfg.device,
            metric_logger=metric_logger,
        )
        _install_expert_episodes(
            irl_alg,
            train_cfg.irl.expert_data_path,
            expected_feature_dim=runtime_ctx.feature_dim,
            expected_feature_names=feature_names,
            max_num_trajectories=train_cfg.irl.expert_num_trajectories,
            subset_strategy=train_cfg.irl.expert_subset_strategy,
            seed=train_cfg.seed,
        )
        if train_cfg.resume:
            resume_path = runtime_deps.get_checkpoint_path(
                log_root_path,
                train_cfg.load_run,
                train_cfg.load_checkpoint,
            )
            runner.load(resume_path, load_optimizer=True)
        _dump_run_configs(
            log_dir=log_dir, env_cfg=env_cfg, train_cfg=train_cfg, env=env
        )
        runner.learn(
            num_learning_iterations=int(train_cfg.max_iterations),
            init_at_random_ep_len=True,
        )
        if (
            args_cli.video
            and args_cli.video_final_only
            and video.record_video_wrapper is not None
        ):
            # Explicit post-training rollout drives the final clip(s); reliable
            # even in BC-only mode where the env isn't stepped during training.
            try:
                runner.record_final_rollouts(
                    recorder=video.record_video_wrapper,
                    video_length=int(args_cli.video_length),
                )
            except Exception as exc:
                print(f"[WARN] Failed to record final rollout videos: {exc}")
    except BaseException:
        # SimulationApp.close() in the finally below calls os._exit(0) which
        # terminates before Python flushes its default traceback. Print
        # explicitly so silent crashes show up in the sweep logs.
        run_crashed = True
        import traceback

        print("[ERROR] train_irl.py crashed:", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception as exc:
                print(f"[WARN] Failed to close environment cleanly: {exc}")
        if (
            video.video_dir_for_iteration_names is not None
            and video.video_steps_per_iteration is not None
        ):
            try:
                rename_iteration_files(
                    video_dir=video.video_dir_for_iteration_names,
                    steps_per_iteration=video.video_steps_per_iteration,
                )
            except Exception as exc:
                print(
                    f"[WARN] Failed to rename video files with iteration indices: {exc}"
                )
        if log_dir is not None:
            copy_final_videos(args_cli, log_dir=log_dir, video=video)
        if runner is not None:
            # Explicit finalize: SimulationApp.close() below calls os._exit(0),
            # which skips wandb's atexit hook and leaves the run marked as
            # crashed on the server. Call wandb.finish() here so completed
            # runs report the correct state.
            runner.finish_logger(exit_code=1 if run_crashed else 0)
        simulation_app.close()


if __name__ == "__main__":
    main()
