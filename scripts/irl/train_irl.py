# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train IRL agent with PPO + feature-buffer IRL."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path

# Load .env for WANDB_API_KEY before any wandb imports
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_path)

from typing import Any, Callable

import torch
import yaml

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cli_args import build_arg_parser

from adapters import RslRlVecEnvAdapter
from algorithms import IrlAlgorithm
from algorithms.bc import BC
from algorithms.ppo_with_bc import make_ppo_with_bc_cls
from config import ActorCriticCfg, TrainCfg, load_train_cfg
from data import (
    load_expert_success_rate,
    load_feature_episodes,
    load_obs_action_episodes,
    split_holdout_episodes,
    subset_episodes,
)
from reward_features.manager_based import (
    ManagerBasedFeatureCfg,
    ManagerBasedFeatureSource,
)
from reward_features.success_bonus import add_success_bonus_term
from reward_model import DenseFeatureRewardModel, LinearFeatureRewardModel
from runner import IrlRunner, NoopMetricLogger, RslRlPpoAdapter, WandbMetricLogger
from utils.runtime_context import RuntimeContext
from storage.obs_action_storage import ObsActionBufCfg, ObsActionBuffer


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]



def _build_actor_critic(
    *,
    obs: Any,
    obs_groups: dict[str, list[str]],
    action_dim: int,
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
        int(action_dim),
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
        from isaaclab_tasks.manager_based.manipulation.lift.mdp import object_reached_goal
    except Exception as exc:
        print(f"[WARN] No success metric for task {task_name!r} (object_reached_goal unavailable): {exc}")
        return None

    def success_fn(env: Any) -> torch.Tensor:
        return object_reached_goal(env.unwrapped, threshold=threshold)

    try:
        probe = success_fn(probe_env)
        if not isinstance(probe, torch.Tensor):
            raise TypeError(f"object_reached_goal returned {type(probe)}, expected Tensor.")
    except Exception as exc:
        print(f"[WARN] Disabling success metric for task {task_name!r} (probe failed): {exc}")
        return None

    print(f"[INFO] Success metric enabled: object within {threshold} m of goal pose.")
    return success_fn



def _install_expert_episodes(
    irl_alg: IrlAlgorithm,
    expert_data_path: str,
    *,
    expected_feature_dim: int,
    expected_feature_names: list[str] | None = None,
    max_num_trajectories: int | None = None,
    subset_strategy: str = "first",
    seed: int = 42,
) -> None:
    """Load expert feature episodes from disk and add them to ``irl_alg``.

    The expert buffer is initialized eagerly by ``IrlAlgorithm.__init__`` with
    ``expert_num_envs`` envs; this helper appends one episode per demo.
    """
    expert_ctx = irl_alg.expert_storage.ctx
    if int(expert_ctx.num_envs) != 1:
        raise ValueError(
            "Expert loader requires IrlAlgorithm.expert_storage with num_envs == 1, "
            f"got {int(expert_ctx.num_envs)}."
        )

    episodes = load_feature_episodes(
        os.path.abspath(expert_data_path),
        expected_feature_dim=expected_feature_dim,
        expected_feature_names=expected_feature_names,
    )
    total = len(episodes)
    if max_num_trajectories is not None:
        episodes = subset_episodes(episodes, max_num_trajectories, subset_strategy, seed)
        print(f"[INFO] Expert subset: {len(episodes)}/{total} trajectories (strategy={subset_strategy})")
    for episode in episodes:
        irl_alg.add_expert_episode(episode)


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
        return {str(name): float(getattr(cfg, "weight", 0.0)) for name, cfg in zip(term_names, term_cfgs)}
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
            _dump_yaml(os.path.join(params_dir, "ground_truth_reward_weights.yaml"), ground_truth)
            weights_str = "  ".join(f"{k}={v:.4f}" for k, v in ground_truth.items())
            print(f"[INFO] Ground truth reward weights: {weights_str}")


def _latest_mp4_in(video_dir: str) -> str | None:
    """Highest-numbered ``*-iter-K.mp4`` (or ``*-step-N.mp4``) in ``video_dir``."""
    if not os.path.isdir(video_dir):
        return None
    mp4s = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not mp4s:
        return None
    pat = re.compile(r"-(?:iter|step)-(\d+)\.mp4$")
    latest = max(mp4s, key=lambda name: (int(pat.search(name).group(1)) if pat.search(name) else -1, name))
    return os.path.join(video_dir, latest)


def _build_video_step_trigger(
    *,
    video_final_only: bool,
    video_interval_iterations: int | None,
    video_interval: int,
    max_iterations: int,
    steps_per_iter: int,
    train_video_dir: str,
) -> tuple[Callable[[int], bool], str | None, int | None]:
    """Build the gym ``RecordVideo.step_trigger`` for the requested mode.

    Returns ``(step_trigger, rename_dir, steps_per_iter)`` where the last two
    are non-None only for the iteration-aligned modes (final-only and
    interval-by-iterations). For those modes, the caller post-processes
    ``rename_dir`` to rewrite ``*-step-N.mp4`` filenames to ``*-iter-K.mp4``.
    The pure-step interval mode returns ``(trigger, None, None)``.
    """
    if video_final_only:
        # Trigger fires at the start of the last iteration so the clip captures
        # the final rollout. `video_length` may exceed `steps_per_iter`; gym's
        # RecordVideo keeps recording across env.step() calls.
        final_trigger_step = max(0, (max_iterations - 1) * steps_per_iter)
        return (lambda step_idx: step_idx == final_trigger_step), train_video_dir, steps_per_iter

    if video_interval_iterations is not None:
        if video_interval_iterations <= 0:
            raise ValueError(
                f"`--video_interval_iterations` must be > 0, got {video_interval_iterations}."
            )
        interval_steps = video_interval_iterations * steps_per_iter
        return (lambda step_idx: step_idx % interval_steps == 0), train_video_dir, steps_per_iter

    if video_interval <= 0:
        raise ValueError(f"`--video_interval` must be > 0, got {video_interval}.")
    return (lambda step_idx: step_idx % video_interval == 0), None, None


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
    obs_groups = resolve_obs_groups_fn(obs, {}, default_sets=["actor", "critic"])
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




def main(argv: list[str] | None = None, deps: RuntimeDeps | None = None) -> None:
    runtime_deps = deps
    if runtime_deps is None:
        app_launcher_cls = _load_app_launcher_cls()
    else:
        app_launcher_cls = runtime_deps.AppLauncher

    parser = build_arg_parser(app_launcher_cls)
    args_cli = parser.parse_args(argv)
    if args_cli.video:
        args_cli.enable_cameras = True

    app_launcher = app_launcher_cls(args_cli)
    simulation_app = app_launcher.app
    env = None
    runner: IrlRunner | None = None
    log_dir: str | None = None
    video_dir_for_iteration_names: str | None = None
    video_steps_per_iteration: int | None = None
    record_video_wrapper: Any = None
    train_video_dir: str | None = None
    run_crashed = False

    if runtime_deps is None:
        # Isaac/Omniverse runtime deps must be imported only after SimulationApp creation.
        runtime_deps = _load_runtime_deps(app_launcher_cls=app_launcher_cls)

    try:
        requested_task_name = args_cli.task
        if requested_task_name is None:
            raise ValueError("`--task` is required.")

        train_cfg = load_train_cfg(
            task_name=requested_task_name,
            args_cli=args_cli,
            config_dir=_repo_root() / "configs" / "franka_lift",
        )

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
                print("[WARN] add_success_bonus=True but env_cfg has no `rewards` section; skipped.")

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
            train_video_dir = os.path.join(log_dir, "videos", "train")
            if args_cli.video_final_only:
                # Final clip(s) are produced by an explicit post-training rollout
                # (see runner.record_final_rollouts), not by an in-training
                # trigger. Install a never-firing trigger and disable auto-stop
                # (video_length=0) so recording boundaries are fully manual.
                env = runtime_deps.gym.wrappers.RecordVideo(
                    env,
                    video_folder=train_video_dir,
                    step_trigger=lambda step_idx: False,
                    video_length=0,
                    disable_logger=True,
                )
                record_video_wrapper = env
            else:
                step_trigger, video_dir_for_iteration_names, video_steps_per_iteration = (
                    _build_video_step_trigger(
                        video_final_only=False,
                        video_interval_iterations=(
                            int(args_cli.video_interval_iterations)
                            if args_cli.video_interval_iterations is not None
                            else None
                        ),
                        video_interval=int(args_cli.video_interval),
                        max_iterations=int(train_cfg.max_iterations),
                        steps_per_iter=int(train_cfg.runner.steps_per_env_per_cycle),
                        train_video_dir=train_video_dir,
                    )
                )
                env = runtime_deps.gym.wrappers.RecordVideo(
                    env,
                    video_folder=train_video_dir,
                    step_trigger=step_trigger,
                    video_length=args_cli.video_length,
                    disable_logger=True,
                )

        env = runtime_deps.RslRlVecEnvWrapper(env)
        if hasattr(env, "seed"):
            env.seed(int(train_cfg.seed))
        # Wrap once so the rest of the script sees `env.action_dim`; the
        # adapter proxies every other attribute to the rsl_rl VecEnv.
        env = RslRlVecEnvAdapter(env)

        obs, obs_groups = _get_initial_obs(env, runtime_deps.resolve_obs_groups)
        obs = obs.to(train_cfg.device)
        actor, critic = _build_actor_critic(
            obs=obs,
            obs_groups=obs_groups,
            action_dim=int(env.action_dim),
            actor_critic_cfg=train_cfg.actor_critic,
            device=train_cfg.device,
            MLPModel=runtime_deps.MLPModel,
        )

        feature_source = ManagerBasedFeatureSource(
            env, cfg=train_cfg.feature_map, device=train_cfg.device,
        )
        feature_map = feature_source.features_from_env
        runtime_ctx = RuntimeContext(
            num_envs=int(env.num_envs),
            feature_dim=int(feature_source.spec.dim),
            device=train_cfg.device,
        )
        feature_names = list(feature_source.spec.names or ())

        # Optional task success predicate for the validation success-rate metric.
        # Uses the Lift task's built-in `object_reached_goal` (object within
        # `env.success_threshold` of the commanded goal pose). Disabled gracefully
        # for tasks that don't provide it.
        success_fn = _build_success_fn(
            task_name=requested_task_name,
            threshold=float(train_cfg.env.success_threshold),
            probe_env=env,
        )

        bc_alg: BC | None = None
        if train_cfg.bc.alpha > 0.0:
            if not train_cfg.irl.expert_data_path:
                raise ValueError(
                    "`bc.alpha > 0` requires `irl.expert_data_path` to point at an "
                    "HDF5 file with (obs, actions) demos."
                )
            obs_action_episodes = load_obs_action_episodes(train_cfg.irl.expert_data_path)
            if train_cfg.irl.expert_num_trajectories is not None:
                total = len(obs_action_episodes)
                obs_action_episodes = subset_episodes(
                    obs_action_episodes,
                    int(train_cfg.irl.expert_num_trajectories),
                    train_cfg.irl.expert_subset_strategy,
                    train_cfg.seed,
                )
                print(
                    f"[INFO] BC expert subset: {len(obs_action_episodes)}/{total} "
                    f"trajectories (strategy={train_cfg.irl.expert_subset_strategy})"
                )
            train_episodes, val_episodes = split_holdout_episodes(
                obs_action_episodes, train_cfg.bc.val_fraction, train_cfg.seed
            )
            obs_action_storage = ObsActionBuffer(
                cfg=ObsActionBufCfg(store_device="cpu", store_dtype=torch.float32),
                ctx=runtime_ctx,
            )
            obs_action_storage.load_episodes(train_episodes)
            val_storage: ObsActionBuffer | None = None
            if val_episodes:
                val_storage = ObsActionBuffer(
                    cfg=ObsActionBufCfg(store_device="cpu", store_dtype=torch.float32),
                    ctx=runtime_ctx,
                )
                val_storage.load_episodes(val_episodes)
            print(
                f"[INFO] BC dataset: {len(train_episodes)} train episodes "
                f"({len(obs_action_storage)} pairs)"
                + (
                    f", {len(val_episodes)} val episodes ({len(val_storage)} pairs)"
                    if val_storage is not None
                    else " (no validation split)"
                )
                + "."
            )
            bc_alg = BC(
                cfg=train_cfg.bc,
                storage=obs_action_storage,
                val_storage=val_storage,
                device=train_cfg.device,
            )

        rollout_storage = runtime_deps.RolloutStorage(
            "rl",
            int(env.num_envs),
            int(train_cfg.runner.steps_per_env_per_cycle),
            obs,
            [int(env.action_dim)],
            train_cfg.device,
        )
        ppo_kwargs = asdict(train_cfg.rl_algorithm)
        if bc_alg is not None:
            ppo_cls = make_ppo_with_bc_cls(runtime_deps.PPO)
            ppo_alg = ppo_cls(
                actor=actor,
                critic=critic,
                storage=rollout_storage,
                device=train_cfg.device,
                bc_alg=bc_alg,
                **ppo_kwargs,
            )
        else:
            ppo_alg = runtime_deps.PPO(
                actor=actor,
                critic=critic,
                storage=rollout_storage,
                device=train_cfg.device,
                **ppo_kwargs,
            )
        reward_cfg = replace(train_cfg.reward, num_features=runtime_ctx.feature_dim)
        reward_model_cls = LinearFeatureRewardModel if reward_cfg.is_linear else DenseFeatureRewardModel
        reward_model = reward_model_cls(reward_cfg).to(train_cfg.device)
        rl_alg = RslRlPpoAdapter(
            ppo_alg,
            env=env,
            feature_map=feature_map,
            reward_model=reward_model,
            steps_per_env_per_cycle=int(train_cfg.runner.steps_per_env_per_cycle),
            use_learned_reward=bool(train_cfg.runner.use_learned_reward),
            device=train_cfg.device,
            bc_alg=bc_alg,
        )
        expert_success_rate: float | None = None
        if train_cfg.irl.expert_data_path:
            try:
                expert_success_rate = load_expert_success_rate(
                    os.path.abspath(train_cfg.irl.expert_data_path)
                )
            except Exception as exc:
                print(f"[WARN] Could not read expert success rate from demos: {exc}")
                expert_success_rate = None
            if expert_success_rate is not None:
                print(f"[INFO] Expert success rate (from demos): {expert_success_rate:.4f}")
            else:
                print("[INFO] No expert success info in demos; expert-success gap disabled.")
        wandb_config = {
            "task": str(train_cfg.env.name),
            "num_envs": int(env.num_envs),
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
            "ppo/learning_rate": float(train_cfg.rl_algorithm.learning_rate),
            "ppo/gamma": float(train_cfg.rl_algorithm.gamma),
            "ppo/lam": float(train_cfg.rl_algorithm.lam),
            "reward/type": "linear" if train_cfg.reward.is_linear else "dense",
            "reward/regularization": str(train_cfg.reward.regularization),
            "reward/regularization_strength": float(train_cfg.reward.regularization_strength),
            "runner/reward_updates_per_cycle": int(train_cfg.runner.reward_updates_per_cycle),
            "runner/rl_updates_per_cycle": int(train_cfg.runner.rl_updates_per_cycle),
        }
        wandb_tags = [
            f"alpha={train_cfg.bc.alpha}",
            f"seed={train_cfg.seed}",
            f"loss={train_cfg.bc.loss_type}",
        ]

        if train_cfg.runner.use_learned_reward:
            print("[INFO] IRL mode: agent trains against the learned reward model.")
        else:
            print(
                "[INFO] IRL mode: agent trains against the env's GT reward "
                "(learned reward model fit as diagnostic only)."
            )

        irl_alg = IrlAlgorithm(
            reward=reward_model,
            feature_spec=feature_source.spec,
            num_envs=int(env.num_envs),
            gamma=float(train_cfg.rl_algorithm.gamma),
            cfg=train_cfg.irl,
            device=train_cfg.device,
            imitator_buffer_cfg=train_cfg.runner.imitator_buffer,
            expert_buffer_cfg=train_cfg.runner.expert_buffer,
            expert_num_envs=int(train_cfg.runner.expert_num_envs),
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
        if train_cfg.irl.expert_data_path:
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
        _dump_run_configs(log_dir=log_dir, env_cfg=env_cfg, train_cfg=train_cfg, env=env)
        runner.learn(
            num_learning_iterations=int(train_cfg.max_iterations),
            init_at_random_ep_len=True,
        )
        if args_cli.video and args_cli.video_final_only and record_video_wrapper is not None:
            # Explicit post-training rollout drives the final clip(s); reliable
            # even in BC-only mode where the env isn't stepped during training.
            try:
                runner.record_final_rollouts(
                    recorder=record_video_wrapper,
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
        if video_dir_for_iteration_names is not None and video_steps_per_iteration is not None:
            try:
                _rename_video_files_to_iteration_index(
                    video_dir=video_dir_for_iteration_names,
                    steps_per_iteration=video_steps_per_iteration,
                )
            except Exception as exc:
                print(f"[WARN] Failed to rename video files with iteration indices: {exc}")
        if args_cli.video and log_dir is not None and train_video_dir is not None:
            videos_dir = os.path.join(log_dir, "videos")
            if args_cli.video_final_only:
                # Explicit dual-clip mode: deterministic + stochastic, named at
                # the source by runner.record_final_rollouts.
                for mode_name in ("inference", "stochastic"):
                    source = os.path.join(train_video_dir, f"final_{mode_name}.mp4")
                    if not os.path.isfile(source):
                        continue
                    final_path = os.path.join(videos_dir, f"final_{mode_name}.mp4")
                    try:
                        shutil.copy2(source, final_path)
                        print(f"[INFO] Copied final {mode_name} rollout video to {final_path}")
                    except Exception as exc:
                        print(f"[WARN] Failed to copy final {mode_name} video locally: {exc}")
                        continue
            else:
                # Interval modes: copy the latest trigger-produced clip.
                try:
                    latest_mp4 = _latest_mp4_in(train_video_dir)
                    if latest_mp4 is not None:
                        final_path = os.path.join(videos_dir, "final.mp4")
                        shutil.copy2(latest_mp4, final_path)
                        print(f"[INFO] Copied final rollout video to {final_path}")
                except Exception as exc:
                    print(f"[WARN] Failed to copy final video locally: {exc}")
        if runner is not None:
            # Explicit finalize: SimulationApp.close() below calls os._exit(0),
            # which skips wandb's atexit hook and leaves the run marked as
            # crashed on the server. Call wandb.finish() here so completed
            # runs report the correct state.
            runner.finish_logger(exit_code=1 if run_crashed else 0)
        simulation_app.close()


if __name__ == "__main__":
    main()
