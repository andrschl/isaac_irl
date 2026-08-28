"""Isaac construction boundary for the fixed-reward PPO/SAC/PMD benchmark."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import torch

from adapters.rsl_rl_actor_critic import make_actor_critic
from adapters.rsl_rl_fixed_reward_ppo import FixedRewardPpoAdapter
from adapters.rsl_rl_off_policy_factory import build_off_policy_stack
from adapters.rsl_rl_vec_env import RslRlVecEnvAdapter
from config.benchmark import BenchmarkTask, FixedRewardBenchmarkCfg
from config.fixed_reward import FixedRewardRunnerCfg
from config.train_irl import ActorCriticCfg
from runner import CompositeLogger, ConsoleLogger, FixedRewardRunner, WandbLogger


@dataclass(slots=True)
class BuiltBenchmark:
    env: Any
    runner: FixedRewardRunner
    log_dir: str
    video_recorder: Any | None
    training_video_dir: str | None


def build_fixed_reward_benchmark(cfg: FixedRewardBenchmarkCfg) -> BuiltBenchmark:
    """Construct a seeded environment and one learner for the chosen task."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import parse_env_cfg

    from reward_features.lift_success import add_success_bonus_term, build_success_fn

    task = cfg.task
    torch.manual_seed(cfg.seed)
    num_envs = _environment_count(cfg)
    env_cfg = parse_env_cfg(
        task.task_id,
        device=cfg.device,
        num_envs=num_envs,
        use_fabric=True,
    )
    env_cfg.seed = cfg.seed
    if task.add_success_bonus:
        if task.success_threshold is None:
            raise ValueError("add_success_bonus requires a success_threshold.")
        # Weight zero: a feature for the critic and the reward learner, never
        # part of the environment reward the benchmark trains against.
        add_success_bonus_term(env_cfg, threshold=task.success_threshold, weight=0.0)

    run_name = cfg.run_name or f"{cfg.algorithm}-seed-{cfg.seed}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(
        os.path.join(cfg.log_root, cfg.group, f"{timestamp}_{run_name}")
    )
    os.makedirs(log_dir, exist_ok=True)

    native_env = gym.make(
        task.task_id,
        cfg=env_cfg,
        render_mode="rgb_array" if cfg.record_video else None,
    )
    video_recorder = None
    training_video_dir = None
    if cfg.record_video:
        training_video_dir = os.path.join(log_dir, "videos", "train")
        native_env = gym.wrappers.RecordVideo(
            native_env,
            video_folder=training_video_dir,
            step_trigger=lambda _: False,
            video_length=0,
            disable_logger=True,
        )
        video_recorder = native_env

    # No environment clipping: the tanh actors are bounded by construction and
    # the on-policy baseline keeps the action space its tuning assumes.
    env = RslRlVecEnvAdapter(RslRlVecEnvWrapper(native_env, clip_actions=None))
    env.seed(cfg.seed)
    observations = env.get_observations()
    success_fn = (
        build_success_fn(
            task_name=task.task_id,
            threshold=task.success_threshold,
            probe_env=env,
        )
        if task.success_threshold is not None
        else None
    )

    if cfg.algorithm == "ppo":
        algorithm, algorithm_config = _build_ppo(cfg, observations, env)
    else:
        algorithm, algorithm_config = _build_off_policy(cfg, observations, env)
    variant = (
        "ppo"
        if cfg.algorithm == "ppo"
        else f"{cfg.algorithm}-{algorithm_config['value_backend']}"
    )

    logger = CompositeLogger(
        [
            WandbLogger(
                project=cfg.project,
                log_dir=log_dir,
                run_name=run_name,
                group=cfg.group,
                tags=[
                    task.name,
                    "fixed_reward",
                    "sample_efficiency",
                    cfg.algorithm,
                    variant,
                    f"seed={cfg.seed}",
                ],
                config={
                    "algorithm": cfg.algorithm,
                    "seed": cfg.seed,
                    "task": task.task_id,
                    "task_profile": task.name,
                    "num_envs": num_envs,
                    "transition_budget": task.transition_budget,
                    "evaluation_steps": task.evaluation_steps,
                    "variant": variant,
                    "bc/enabled": False,
                    "bc/weight": 0.0,
                    "reward/fixed_environment_reward": True,
                    "reward/learnable": False,
                    "algorithm/config": algorithm_config,
                },
            ),
            ConsoleLogger(),
        ]
    )
    runner = FixedRewardRunner(
        env=env,
        rl_algorithm=algorithm,
        cfg=FixedRewardRunnerCfg(
            save_interval=50,
            validation_interval=0,
            validation_steps=task.evaluation_steps,
            validate_before_training=True,
            validation_interval_transitions=task.evaluation_interval_transitions,
            return_per_step_thresholds=task.return_per_step_thresholds,
            success_rate_thresholds=task.success_rate_thresholds,
        ),
        device=cfg.device,
        log_dir=log_dir,
        success_fn=success_fn,
        logger=logger,
    )
    return BuiltBenchmark(
        env=env,
        runner=runner,
        log_dir=log_dir,
        video_recorder=video_recorder,
        training_video_dir=training_video_dir,
    )


def _environment_count(cfg: FixedRewardBenchmarkCfg) -> int:
    """Environments to simulate: PPO may run at its own tuned parallelism."""
    if cfg.algorithm == "ppo" and cfg.task.ppo.num_envs is not None:
        return int(cfg.task.ppo.num_envs)
    return int(cfg.task.num_envs)


def _build_ppo(
    cfg: FixedRewardBenchmarkCfg, observations: Any, env: Any
) -> tuple[Any, dict[str, Any]]:
    """Build the tuned rsl-rl PPO baseline for this task."""
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from rsl_rl.utils import resolve_obs_groups

    task: BenchmarkTask = cfg.task
    baseline = task.ppo
    obs_groups = resolve_obs_groups(
        observations,
        {"actor": ["policy"], "critic": ["policy"]},
        default_sets=["actor", "critic"],
    )
    actor, critic = make_actor_critic(
        obs=observations,
        obs_groups=obs_groups,
        action_dim=env.action_dim,
        cfg=ActorCriticCfg(
            actor_hidden_dims=baseline.hidden_dims,
            critic_hidden_dims=baseline.hidden_dims,
            activation="elu",
            init_noise_std=baseline.init_noise_std,
        ),
        device=cfg.device,
        MLPModel=MLPModel,
    )
    storage = RolloutStorage(
        "rl",
        _environment_count(cfg),
        baseline.rollout_steps,
        observations,
        [env.action_dim],
        cfg.device,
    )
    ppo = PPO(
        actor=actor,
        critic=critic,
        storage=storage,
        num_learning_epochs=baseline.num_learning_epochs,
        num_mini_batches=baseline.num_mini_batches,
        clip_param=baseline.clip_param,
        gamma=baseline.gamma,
        lam=baseline.lam,
        value_loss_coef=baseline.value_loss_coef,
        entropy_coef=baseline.entropy_coef,
        learning_rate=baseline.learning_rate,
        max_grad_norm=baseline.max_grad_norm,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=baseline.desired_kl,
        device=cfg.device,
    )
    algorithm = FixedRewardPpoAdapter(
        env=env,
        algorithm=ppo,
        steps_per_env=baseline.rollout_steps,
        device=cfg.device,
    )
    return algorithm, {"source": "rsl_rl.PPO", "value_backend": "ppo"} | asdict(
        baseline
    )


def _build_off_policy(
    cfg: FixedRewardBenchmarkCfg, observations: Any, env: Any
) -> tuple[Any, dict[str, Any]]:
    """Build SAC or PMD with the task's feature space and value backend."""
    task = cfg.task
    algorithm_cfg = task.sac if cfg.algorithm == "sac" else task.pmd
    stack = build_off_policy_stack(
        env=env,
        observations=observations,
        algorithm_cfg=algorithm_cfg,
        ignored_reward_terms=set(task.ignored_reward_terms),
        force_include_reward_terms=set(task.force_include_reward_terms),
        device=cfg.device,
        seed=cfg.seed,
    )
    algorithm_config = asdict(algorithm_cfg) | {
        "source": f"local.{cfg.algorithm.upper()}",
        "value_backend": algorithm_cfg.value.type,
    }
    return stack.algorithm, algorithm_config


def record_and_promote_video(built: BuiltBenchmark, video_length: int) -> None:
    """Record one deterministic final rollout and promote it for W&B upload."""
    if built.video_recorder is None or built.training_video_dir is None:
        return
    built.runner.record_final_rollouts(
        recorder=built.video_recorder,
        video_length=video_length,
        mode_names=("inference",),
    )
    source = os.path.join(built.training_video_dir, "final_inference.mp4")
    if os.path.isfile(source):
        destination_dir = os.path.join(built.log_dir, "videos")
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copy2(source, os.path.join(destination_dir, "final_inference.mp4"))
