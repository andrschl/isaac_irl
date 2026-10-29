"""Run one training job: build everything, learn, and shut Isaac down cleanly."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

from adapters.pmd import PmdAdapter
from config import build_arg_parser, load_experiment_cfg
from config.schema import PpoAlgoCfg, TrainCfg
from interfaces import RuntimeContext
from runner import CompositeLogger, ConsoleLogger, LogEvent, Runner, WandbLogger
from training import build
from training.run_output import dump_run_configs
from training.video import build_video_step_trigger, finalize_videos

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class RuntimeDeps:
    """Isaac and rsl_rl objects, imported only after the simulator app exists."""

    AppLauncher: Any
    gym: Any
    parse_env_cfg: Callable[..., Any]
    get_checkpoint_path: Callable[..., str]
    RslRlVecEnvWrapper: Any
    PPO: Any
    MLPModel: Any
    RolloutStorage: Any
    resolve_obs_groups: Callable[..., Any]


@dataclass
class _RunState:
    """What the lifecycle needs to clean up, filled in as it is built."""

    env: Any = None
    runner: Runner | None = None
    log_dir: str | None = None
    train_video_dir: str | None = None
    video_rename_dir: str | None = None
    video_steps_per_iteration: int | None = None


def load_runtime_deps(app_launcher_cls: Any) -> RuntimeDeps:
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
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


@contextmanager
def isaac_lifecycle(simulation_app: Any, args: argparse.Namespace):
    """Own the simulator app, env close, logger finalization, and video copy.

    Kit owns the process return code once ``SimulationApp.close()`` begins, so
    failures publish a nonzero return code before the graceful shutdown.
    """
    state = _RunState()
    crashed = False
    try:
        yield state
    except BaseException:
        crashed = True
        import traceback

        print("[ERROR] training crashed:", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
    finally:
        if state.env is not None:
            state.env.close()
        finalize_videos(
            args,
            log_dir=state.log_dir,
            train_video_dir=state.train_video_dir,
            rename_dir=state.video_rename_dir,
            rename_steps_per_iter=state.video_steps_per_iteration,
        )
        if state.runner is not None:
            state.runner.finish_logger(exit_code=1 if crashed else 0)
        if crashed:
            simulation_app.app.post_quit(return_code=1)
        simulation_app.close()


def _log_dir(cfg: TrainCfg) -> tuple[str, str]:
    root = os.path.abspath(os.path.join("logs", "train", cfg.experiment_name))
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.run_name:
        name += f"_{cfg.run_name}"
    directory = os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)
    return root, directory


def _build_logger(cfg: TrainCfg, *, log_dir: str, env: Any) -> CompositeLogger:
    sinks: list[Any] = [ConsoleLogger()]
    if cfg.logger == "wandb":
        sinks.insert(
            0,
            WandbLogger(
                project=cfg.wandb_project,
                log_dir=log_dir,
                run_name=cfg.run_name or None,
                group=cfg.experiment_name,
                tags=[
                    f"algorithm={cfg.algorithm.type}",
                    f"reward={cfg.reward.source}",
                    f"bc_weight={cfg.bc.weight}",
                    f"seed={cfg.seed}",
                ],
                config=_wandb_config(cfg, env=env),
            ),
        )
    return CompositeLogger(sinks)


def _log_run_facts(
    logger: Any, *, cfg: TrainCfg, reward_learner: Any, env: Any
) -> None:
    """Record the setup numbers a finished run has to be read against."""
    payload: dict[str, float] = {
        "Setup/num_envs": float(env.num_envs),
        "Setup/bc_weight": float(cfg.bc.weight),
    }
    if reward_learner is not None:
        payload["Setup/expert_episodes"] = float(len(reward_learner.expert_storage))
        if reward_learner.expert_success_rate is not None:
            payload["Summary/expert_success_rate"] = float(
                reward_learner.expert_success_rate
            )
    logger.log(LogEvent(kind="setup", step=0, payload=payload))


def _wandb_config(cfg: TrainCfg, *, env: Any) -> dict[str, Any]:
    value_backend = (
        None if isinstance(cfg.algorithm, PpoAlgoCfg) else cfg.algorithm.value.type
    )
    return {
        "task": str(cfg.env.name),
        "num_envs": int(env.num_envs),
        "seed": int(cfg.seed),
        "max_iterations": int(cfg.max_iterations),
        "algorithm": cfg.algorithm.type,
        "value_backend": value_backend,
        "gamma": cfg.gamma,
        "env/reward_curriculum": cfg.env.reward_curriculum,
        "env/variant": cfg.env.variant,
        "env/lift_cube_xy_half_range": cfg.env.lift_cube_xy_half_range,
        "runner/total_transition_budget": cfg.runner.total_transition_budget,
        "pmd/episodes_per_batch": None
        if cfg.is_on_policy
        else cfg.algorithm.train.episodes_per_batch,
        "pmd/critic_updates_per_batch": None
        if cfg.is_on_policy
        else cfg.algorithm.train.critic_updates_per_batch,
        "pmd/critic_updates_per_transition": None
        if cfg.is_on_policy
        else cfg.algorithm.train.critic_updates_per_transition,
        "pmd/policy_improvements_per_batch": None
        if cfg.is_on_policy
        else cfg.algorithm.policy_improvements_per_batch,
        "reward/source": cfg.reward.source,
        "reward/environment_weight": cfg.reward.environment_weight,
        "reward/learned_weight": cfg.reward.learned_weight,
        "reward/type": "linear" if cfg.reward.model.is_linear else "dense",
        "reward/regularization": cfg.reward.model.regularization,
        "reward/learned_feature_scales": cfg.reward.learned_feature_scales,
        "bc/weight": float(cfg.bc.weight),
        "bc/loss_type": cfg.bc.loss_type,
        "irl/expert_data_path": cfg.expert_data.path,
        "irl/expert_num_trajectories": cfg.expert_data.num_trajectories,
        "irl/reward_learning_rate": cfg.irl.reward_learning_rate,
        "irl/solver": cfg.irl.solver,
        "runner/steps_per_env_per_cycle": cfg.runner.steps_per_env_per_cycle,
        "runner/reward_updates_per_cycle": cfg.runner.reward_updates_per_cycle,
        "runner/imitator_source": cfg.runner.imitator_source,
    }


def _wrap_for_video(
    env: Any, *, cfg: TrainCfg, args: argparse.Namespace, state: _RunState, gym: Any
) -> tuple[Any, Any]:
    """Wrap the env in ``RecordVideo`` and report the recorder, if any."""
    if not args.video:
        return env, None
    state.train_video_dir = os.path.join(str(state.log_dir), "videos", "train")
    if args.video_final_only:
        # The final clips come from an explicit post-training rollout, so the
        # in-training trigger never fires and auto-stop is disabled.
        recorder = gym.wrappers.RecordVideo(
            env,
            video_folder=state.train_video_dir,
            step_trigger=lambda _: False,
            video_length=0,
            disable_logger=True,
        )
        return recorder, recorder
    trigger, state.video_rename_dir, state.video_steps_per_iteration = (
        build_video_step_trigger(
            video_final_only=False,
            video_interval_iterations=args.video_interval_iterations,
            video_interval=2000,
            max_iterations=int(cfg.max_iterations),
            steps_per_iter=int(cfg.runner.steps_per_env_per_cycle),
            train_video_dir=state.train_video_dir,
        )
    )
    return (
        gym.wrappers.RecordVideo(
            env,
            video_folder=state.train_video_dir,
            step_trigger=trigger,
            video_length=args.video_length,
            disable_logger=True,
        ),
        None,
    )


def main(argv: list[str] | None = None, deps: RuntimeDeps | None = None) -> None:
    """Parse the experiment, build it, and run it to completion."""
    app_launcher_cls = deps.AppLauncher if deps is not None else _app_launcher_cls()
    args = build_arg_parser(app_launcher_cls).parse_args(argv)
    if args.video:
        args.enable_cameras = True

    simulation_app = app_launcher_cls(args).app
    with isaac_lifecycle(simulation_app, args) as state:
        if deps is None:
            # Isaac/Omniverse imports are only valid after the app exists.
            deps = load_runtime_deps(app_launcher_cls)

        cfg, task_name = load_experiment_cfg(
            args=args, config_root=REPO_ROOT / "configs"
        )
        env_cfg = build.build_env_cfg(
            cfg, parse_env_cfg=deps.parse_env_cfg, use_fabric=not args.disable_fabric
        )
        log_root, log_dir = _log_dir(cfg)
        state.log_dir = log_dir

        env = deps.gym.make(
            task_name, cfg=env_cfg, render_mode="rgb_array" if args.video else None
        )
        state.env = env
        env, recorder = _wrap_for_video(
            env, cfg=cfg, args=args, state=state, gym=deps.gym
        )
        env = build.wrap_env(
            env, RslRlVecEnvWrapper=deps.RslRlVecEnvWrapper, seed=cfg.seed
        )
        state.env = env

        observations = env.get_observations().to(cfg.device)
        obs_groups = deps.resolve_obs_groups(
            observations, {}, default_sets=["actor", "critic"]
        )
        feature_source = build.build_feature_source(cfg, env=env)
        runtime_ctx = RuntimeContext(
            num_envs=int(env.num_envs),
            feature_dim=int(feature_source.spec.dim),
            device=cfg.device,
        )

        reward_model = build.build_reward_model(
            cfg, env=env, feature_source=feature_source, device=cfg.device
        )
        reward_scorer = build.build_reward_scorer(
            cfg, env=env, feature_source=feature_source, reward_model=reward_model
        )
        bc_alg = build.build_bc(cfg, ctx=runtime_ctx)
        rl_alg = build.build_algorithm(
            cfg,
            env=env,
            observations=observations,
            obs_groups=obs_groups,
            feature_source=feature_source,
            reward=reward_scorer,
            bc_alg=bc_alg,
            deps=deps,
        )
        reward_learner = build.build_reward_learner(
            cfg, env=env, feature_source=feature_source, reward_model=reward_model
        )
        logger = _build_logger(cfg, log_dir=log_dir, env=env)
        runner = build.build_runner(
            cfg,
            env=env,
            rl_alg=rl_alg,
            reward_learner=reward_learner,
            feature_source=feature_source,
            log_dir=log_dir,
            logger=logger,
        )
        state.runner = runner
        _log_run_facts(logger, cfg=cfg, reward_learner=reward_learner, env=env)

        if cfg.resume:
            runner.load(
                deps.get_checkpoint_path(log_root, cfg.load_run, cfg.load_checkpoint),
                load_optimizer=True,
            )
        dump_run_configs(log_dir=log_dir, env_cfg=env_cfg, train_cfg=cfg, env=env)
        if (
            not cfg.resume
            and isinstance(rl_alg, PmdAdapter)
            and rl_alg.replay.demonstration_count
        ):
            warmup_metrics = rl_alg.warm_start_from_demonstrations()
            logger.log(
                LogEvent(
                    kind="initialization",
                    step=0,
                    payload={
                        "Summary/transitions": 0.0,
                        "Summary/environment_transitions": 0.0,
                        **warmup_metrics,
                    },
                )
            )
            runner.save(os.path.join(log_dir, "model_0.pt"))
        runner.learn(int(cfg.max_iterations), init_at_random_ep_len=True)
        if recorder is not None:
            runner.record_final_rollouts(
                recorder=recorder, video_length=int(args.video_length)
            )


def _app_launcher_cls() -> Any:
    from isaaclab.app import AppLauncher

    return AppLauncher


def configure_torch_backends() -> None:
    """Match the numerics the tuned baselines were measured with."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
