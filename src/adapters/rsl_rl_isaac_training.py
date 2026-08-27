# ruff: noqa: E402

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train IRL agent with PPO + feature-buffer IRL."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

# Load .env for WANDB_API_KEY before any wandb imports
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_path)

from typing import Any, Callable

import torch

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from scripts.irl.cli_args import build_arg_parser
from scripts._video_helpers import build_video_step_trigger, finalize_videos

from adapters import (
    RslRlPpoAdapter,
    RslRlVecEnvAdapter,
    load_actor_only_checkpoint,
    make_actor_critic,
)
from adapters.rsl_rl_off_policy_factory import build_off_policy_stack
from algorithms import IrlAlgorithm, build_bc_alg
from algorithms.ppo_with_bc import make_ppo_with_bc_cls
from config import PpoAlgoCfg, load_train_cfg
from config.fixed_reward import FixedRewardRunnerCfg
from config.off_policy import PmdCfg, SacCfg
from data import install_expert_features, load_expert_success_rate
from reward_features.lift_success import add_success_bonus_term, build_success_fn
from reward_features.manager_based import ManagerBasedFeatureSource
from reward_model import DenseFeatureRewardModel, LinearFeatureRewardModel
from reward_model.scorer import (
    LearnedLinearRewardScorer,
    LearnedRewardScorer,
)
from runner import (
    CompositeLogger,
    ConsoleLogger,
    FixedRewardRunner,
    IrlRunner,
    WandbLogger,
)
from utils.run_output import dump_run_configs
from utils.run_output import extract_ground_truth_reward_weights
from utils.runtime_context import RuntimeContext

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


def _build_log_dir(experiment_name: str, run_name: str) -> tuple[str, str]:
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


@dataclass
class _RunState:
    """Mutable state shared between the train_irl ``main`` body and the
    Isaac-lifecycle cleanup in :func:`isaac_lifecycle`. Fields are set as
    objects are constructed; the lifecycle reads them on shutdown."""

    env: Any = None
    runner: Any = None
    log_dir: str | None = None
    train_video_dir: str | None = None
    video_dir_for_iteration_names: str | None = None
    video_steps_per_iteration: int | None = None


@contextmanager
def isaac_lifecycle(simulation_app: Any, args_cli: argparse.Namespace):
    """Own Isaac SimulationApp + env close + wandb finalize + video copy.

    SimulationApp.close() in the cleanup calls ``os._exit(0)`` which skips
    Python's default traceback and wandb's atexit hook, so we print the
    traceback explicitly and call ``runner.finish_logger(exit_code=…)``
    ourselves before closing.
    """
    state = _RunState()
    run_crashed = False
    try:
        yield state
    except BaseException:
        run_crashed = True
        import traceback

        print("[ERROR] train_irl.py crashed:", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
    finally:
        if state.env is not None and hasattr(state.env, "close"):
            try:
                state.env.close()
            except Exception as exc:
                print(f"[WARN] Failed to close environment cleanly: {exc}")
        finalize_videos(
            args_cli,
            log_dir=state.log_dir,
            train_video_dir=state.train_video_dir,
            rename_dir=state.video_dir_for_iteration_names,
            rename_steps_per_iter=state.video_steps_per_iteration,
        )
        if state.runner is not None:
            state.runner.finish_logger(exit_code=1 if run_crashed else 0)
        simulation_app.close()


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

    if runtime_deps is None:
        # Isaac/Omniverse runtime deps must be imported only after SimulationApp creation.
        runtime_deps = _load_runtime_deps(app_launcher_cls=app_launcher_cls)

    with isaac_lifecycle(simulation_app, args_cli) as run:
        record_video_wrapper: Any = None
        requested_task_name = args_cli.task
        if requested_task_name is None:
            raise ValueError("`--task` is required.")

        train_cfg = load_train_cfg(
            task_name=requested_task_name,
            args_cli=args_cli,
            config_dir=_repo_root() / "configs" / "franka_lift",
        )
        if (
            not train_cfg.runner.use_ground_truth_reward_weights
            and not train_cfg.irl.expert_data_path
        ):
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
        if isinstance(env_cfg, MutableMapping):
            env_cfg["seed"] = int(train_cfg.seed)
        elif hasattr(env_cfg, "seed"):
            env_cfg.seed = int(train_cfg.seed)
        else:
            raise TypeError(
                "Parsed environment config must expose a writable `seed` field."
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
        run.log_dir = log_dir

        env = runtime_deps.gym.make(
            requested_task_name,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None,
        )
        run.env = env

        if args_cli.video:
            train_video_dir = os.path.join(log_dir, "videos", "train")
            run.train_video_dir = train_video_dir
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
                (
                    step_trigger,
                    run.video_dir_for_iteration_names,
                    run.video_steps_per_iteration,
                ) = build_video_step_trigger(
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
        run.env = env

        observations, obs_groups = _get_initial_obs(
            env, runtime_deps.resolve_obs_groups
        )
        observations = observations.to(train_cfg.device)

        feature_source = ManagerBasedFeatureSource(
            env,
            cfg=train_cfg.feature_map,
            device=train_cfg.device,
        )
        runtime_ctx = RuntimeContext(
            num_envs=int(env.num_envs),
            feature_dim=int(feature_source.spec.dim),
            device=train_cfg.device,
        )
        success_fn = build_success_fn(
            task_name=requested_task_name,
            threshold=float(train_cfg.env.success_threshold),
            probe_env=env,
        )
        bc_alg = build_bc_alg(
            cfg=train_cfg.bc,
            expert_data_path=train_cfg.irl.expert_data_path,
            ctx=runtime_ctx,
            device=train_cfg.device,
            expert_num_trajectories=train_cfg.irl.expert_num_trajectories,
            expert_subset_strategy=train_cfg.irl.expert_subset_strategy,
            seed=train_cfg.seed,
        )
        is_ppo = isinstance(train_cfg.rl_algorithm, PpoAlgoCfg)
        fixed_off_policy = (
            not is_ppo and train_cfg.runner.use_ground_truth_reward_weights
        )
        reward_model: LinearFeatureRewardModel | DenseFeatureRewardModel | None = None
        if not fixed_off_policy:
            reward_cfg = replace(train_cfg.reward, num_features=runtime_ctx.feature_dim)
            reward_model_cls = (
                LinearFeatureRewardModel
                if reward_cfg.is_linear
                else DenseFeatureRewardModel
            )
            reward_model = reward_model_cls(reward_cfg).to(train_cfg.device)
            if train_cfg.runner.use_ground_truth_reward_weights:
                if not isinstance(reward_model, LinearFeatureRewardModel):
                    raise ValueError(
                        "Ground-truth reward weights require reward.type=linear."
                    )
                if feature_source.spec.names is None:
                    raise ValueError(
                        "Ground-truth reward weights require named reward features."
                    )
                ground_truth_weights = extract_ground_truth_reward_weights(env)
                if not ground_truth_weights:
                    raise ValueError(
                        "Environment does not expose reward-manager ground-truth weights."
                    )
                reward_model.set_named_weights(
                    feature_source.spec.names, ground_truth_weights
                )
                resolved_weights = reward_model.reward.weight.detach().squeeze(0)
                print(
                    "[INFO] Fixed ground-truth reward vector: "
                    + ", ".join(
                        f"{name}={resolved_weights[index].item():.6g}"
                        for index, name in enumerate(feature_source.spec.names)
                    )
                )
        if is_ppo:
            if reward_model is None:
                raise RuntimeError("PPO requires a reward model.")
            actor, critic = make_actor_critic(
                obs=observations,
                obs_groups=obs_groups,
                action_dim=int(env.action_dim),
                cfg=train_cfg.actor_critic,
                device=train_cfg.device,
                MLPModel=runtime_deps.MLPModel,
            )
            if train_cfg.init_policy_checkpoint is not None:
                load_actor_only_checkpoint(
                    actor,
                    os.path.abspath(train_cfg.init_policy_checkpoint),
                    device=train_cfg.device,
                )
            rollout_storage = runtime_deps.RolloutStorage(
                "rl",
                int(env.num_envs),
                int(train_cfg.runner.steps_per_env_per_cycle),
                observations,
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
            rl_alg = RslRlPpoAdapter(
                ppo_alg,
                env=env,
                feature_map=feature_source.features,
                reward_model=reward_model,
                steps_per_env_per_cycle=int(train_cfg.runner.steps_per_env_per_cycle),
                use_learned_reward=bool(train_cfg.runner.use_learned_reward),
                device=train_cfg.device,
                bc_alg=bc_alg,
            )
            algorithm_gamma = float(train_cfg.rl_algorithm.gamma)
        else:
            if not isinstance(train_cfg.rl_algorithm, (SacCfg, PmdCfg)):
                raise TypeError(
                    "Expected off-policy algorithm config SacCfg or PmdCfg, "
                    f"got {type(train_cfg.rl_algorithm)!r}."
                )
            reward_scorer = None
            if not train_cfg.runner.use_ground_truth_reward_weights:
                if reward_model is None:
                    raise RuntimeError(
                        "Learned-reward SAC/PMD requires a reward model."
                    )
                if isinstance(reward_model, LinearFeatureRewardModel):
                    if feature_source.spec.names is None:
                        raise ValueError(
                            "Linear successor rewards require named features."
                        )
                    step_dt = float(
                        getattr(getattr(env, "unwrapped", env), "step_dt", 1.0)
                    )
                    reward_scorer = LearnedLinearRewardScorer(
                        reward_model,
                        feature_names=feature_source.spec.names,
                        step_dt=step_dt,
                    )
                else:
                    reward_scorer = LearnedRewardScorer(
                        reward_model,
                        feature_dim=runtime_ctx.feature_dim,
                    )
            off_policy_stack = build_off_policy_stack(
                env=env,
                observations=observations,
                algorithm_cfg=train_cfg.rl_algorithm,
                ignored_reward_terms=train_cfg.feature_map.ignored_reward_terms,
                device=train_cfg.device,
                seed=train_cfg.seed,
                reward=reward_scorer,
                bc=bc_alg,
            )
            if train_cfg.init_policy_checkpoint is not None:
                load_actor_only_checkpoint(
                    off_policy_stack.algorithm.actor.model,
                    os.path.abspath(train_cfg.init_policy_checkpoint),
                    device=train_cfg.device,
                )
            rl_alg = off_policy_stack.algorithm
            algorithm_gamma = float(train_cfg.rl_algorithm.train.gamma)

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
                print(
                    f"[INFO] Expert success rate (from demos): {expert_success_rate:.4f}"
                )
            else:
                print(
                    "[INFO] No expert success info in demos; expert-success gap disabled."
                )
        wandb_config = {
            "task": str(train_cfg.env.name),
            "num_envs": int(env.num_envs),
            "seed": int(train_cfg.seed),
            "max_iterations": int(train_cfg.max_iterations),
            "bc/weight": float(train_cfg.bc.weight),
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
            "runner/use_ground_truth_reward_weights": bool(
                train_cfg.runner.use_ground_truth_reward_weights
            ),
            "algorithm/type": train_cfg.rl_algorithm.type,
            "algorithm/actor_learning_rate": float(
                train_cfg.rl_algorithm.learning_rate
                if is_ppo
                else train_cfg.rl_algorithm.actor_learning_rate
            ),
            "algorithm/gamma": algorithm_gamma,
            "reward/type": "linear" if train_cfg.reward.is_linear else "dense",
            "reward/regularization": str(train_cfg.reward.regularization),
            "reward/regularization_strength": float(
                train_cfg.reward.regularization_strength
            ),
            "runner/reward_updates_per_cycle": int(
                train_cfg.runner.reward_updates_per_cycle
            ),
            "runner/rl_updates_per_cycle": int(train_cfg.runner.rl_updates_per_cycle),
        }
        wandb_tags = [
            f"bc_weight={train_cfg.bc.weight}",
            f"seed={train_cfg.seed}",
            f"loss={train_cfg.bc.loss_type}",
        ]

        if train_cfg.runner.use_ground_truth_reward_weights:
            print(
                "[INFO] Fixed-reward mode: agent trains against extracted ground-truth feature weights; reward updates are disabled."
            )
        elif train_cfg.runner.use_learned_reward:
            print("[INFO] IRL mode: agent trains against the learned reward model.")
        else:
            print(
                "[INFO] IRL mode: agent trains against the env's GT reward "
                "(learned reward model fit as diagnostic only)."
            )

        sinks = [ConsoleLogger()]
        if train_cfg.logger == "wandb":
            sinks.insert(
                0,
                WandbLogger(
                    project=train_cfg.wandb_project,
                    log_dir=log_dir,
                    run_name=(train_cfg.run_name or None),
                    group=train_cfg.experiment_name,
                    tags=wandb_tags,
                    config=wandb_config,
                ),
            )
        logger = CompositeLogger(sinks)

        if fixed_off_policy:
            validation_steps = train_cfg.runner.validation_steps_per_env
            if validation_steps is None:
                validation_steps = 5 * train_cfg.runner.steps_per_env_per_cycle
            runner = FixedRewardRunner(
                env=env,
                rl_algorithm=rl_alg,
                cfg=FixedRewardRunnerCfg(
                    save_interval=train_cfg.runner.save_interval,
                    validation_interval=train_cfg.runner.validation_interval,
                    validation_steps=max(1, int(validation_steps)),
                ),
                device=train_cfg.device,
                log_dir=log_dir,
                success_fn=success_fn,
                logger=logger,
            )
        else:
            if reward_model is None:
                raise RuntimeError("IRL runner requires a reward model.")
            irl_alg = IrlAlgorithm(
                reward=reward_model,
                feature_spec=feature_source.spec,
                num_envs=int(env.num_envs),
                gamma=algorithm_gamma,
                cfg=train_cfg.irl,
                device=train_cfg.device,
                imitator_buffer_cfg=train_cfg.runner.imitator_buffer,
                expert_buffer_cfg=train_cfg.runner.expert_buffer,
                expert_num_envs=int(train_cfg.runner.expert_num_envs),
                expert_success_rate=expert_success_rate,
            )
            runner = IrlRunner(
                env=env,
                rl_alg=rl_alg,
                irl_alg=irl_alg,
                feature_source=feature_source,
                runner_cfg=train_cfg.runner,
                success_fn=success_fn,
                log_dir=log_dir,
                device=train_cfg.device,
                logger=logger,
            )
            if train_cfg.irl.expert_data_path:
                install_expert_features(
                    irl_alg,
                    train_cfg.irl.expert_data_path,
                    max_num_trajectories=train_cfg.irl.expert_num_trajectories,
                    subset_strategy=train_cfg.irl.expert_subset_strategy,
                    seed=train_cfg.seed,
                )
        run.runner = runner
        if train_cfg.resume:
            resume_path = runtime_deps.get_checkpoint_path(
                log_root_path,
                train_cfg.load_run,
                train_cfg.load_checkpoint,
            )
            runner.load(resume_path, load_optimizer=True)
        dump_run_configs(log_dir=log_dir, env_cfg=env_cfg, train_cfg=train_cfg, env=env)
        if fixed_off_policy:
            runner.learn(num_iterations=int(train_cfg.max_iterations))
        else:
            runner.learn(
                num_learning_iterations=int(train_cfg.max_iterations),
                init_at_random_ep_len=True,
            )
        if (
            args_cli.video
            and args_cli.video_final_only
            and record_video_wrapper is not None
        ):
            # Explicit post-training rollout drives the final clip(s); reliable
            # even in BC-only mode where the env isn't stepped during training.
            try:
                runner.record_final_rollouts(
                    recorder=record_video_wrapper,
                    video_length=int(args_cli.video_length),
                )
            except Exception as exc:
                print(f"[WARN] Failed to record final rollout videos: {exc}")


if __name__ == "__main__":
    main()
