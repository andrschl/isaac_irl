"""Isaac construction boundary for the PPO/SAC/PMD Cartpole benchmark."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import torch

from adapters.rsl_rl_cartpole_ppo import FixedRewardPpoAdapter
from adapters.rsl_rl_off_policy_factory import build_off_policy_stack
from adapters.rsl_rl_actor_critic import make_actor_critic
from adapters.rsl_rl_vec_env import RslRlVecEnvAdapter
from config.cartpole_benchmark import CartpoleBenchmarkCfg
from config.fixed_reward import FixedRewardRunnerCfg
from config.train_irl import ActorCriticCfg
from runner import (
    CompositeLogger,
    ConsoleLogger,
    FixedRewardRunner,
    WandbLogger,
)


@dataclass(slots=True)
class BuiltCartpoleBenchmark:
    env: Any
    runner: FixedRewardRunner
    log_dir: str
    video_recorder: Any | None
    training_video_dir: str | None


def build_cartpole_benchmark(
    cfg: CartpoleBenchmarkCfg,
) -> BuiltCartpoleBenchmark:
    """Construct a seeded Cartpole env and one retained learner."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from rsl_rl.utils import resolve_obs_groups

    torch.manual_seed(cfg.seed)
    task_name = "Isaac-Cartpole-v0"
    env_cfg = parse_env_cfg(
        task_name,
        device=cfg.device,
        num_envs=cfg.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = cfg.seed
    run_name = cfg.run_name or f"{cfg.algorithm}-seed-{cfg.seed}"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(
        os.path.join(cfg.log_root, cfg.group, f"{timestamp}_{run_name}")
    )
    os.makedirs(log_dir, exist_ok=True)

    native_env = gym.make(
        task_name,
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

    env = RslRlVecEnvAdapter(
        RslRlVecEnvWrapper(native_env, clip_actions=cfg.action_clip)
    )
    env.seed(cfg.seed)
    observations = env.get_observations()
    obs_groups = resolve_obs_groups(
        observations,
        {"actor": ["policy"], "critic": ["policy"]},
        default_sets=["actor", "critic"],
    )

    algorithm_config: dict[str, Any]
    if cfg.algorithm == "ppo":
        actor_cfg = ActorCriticCfg(
            actor_hidden_dims=(32, 32),
            critic_hidden_dims=(32, 32),
            activation="elu",
            init_noise_std=1.0,
        )
        actor, critic = make_actor_critic(
            obs=observations,
            obs_groups=obs_groups,
            action_dim=env.action_dim,
            cfg=actor_cfg,
            device=cfg.device,
            MLPModel=MLPModel,
        )
        storage = RolloutStorage(
            "rl", cfg.num_envs, 16, observations, [env.action_dim], cfg.device
        )
        ppo = PPO(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=5,
            num_mini_batches=4,
            clip_param=0.2,
            gamma=0.99,
            lam=0.95,
            value_loss_coef=1.0,
            entropy_coef=0.005,
            learning_rate=1.0e-3,
            max_grad_norm=1.0,
            use_clipped_value_loss=True,
            schedule="adaptive",
            desired_kl=0.01,
            device=cfg.device,
        )
        algorithm = FixedRewardPpoAdapter(
            env=env, algorithm=ppo, steps_per_env=16, device=cfg.device
        )
        algorithm_config = {
            "source": "rsl_rl.PPO",
            "rollout_steps": 16,
            "learning_epochs": 5,
            "mini_batches": 4,
            "learning_rate": 1.0e-3,
            "gamma": 0.99,
        }
    else:
        algorithm_cfg = cfg.sac if cfg.algorithm == "sac" else cfg.pmd
        stack = build_off_policy_stack(
            env=env,
            observations=observations,
            algorithm_cfg=algorithm_cfg,
            ignored_reward_terms={"success_bonus"},
            device=cfg.device,
            seed=cfg.seed,
        )
        algorithm = stack.algorithm
        algorithm_config = asdict(algorithm_cfg) | {
            "source": f"local.{cfg.algorithm.upper()}",
            "value_backend": algorithm_cfg.value.type,
        }
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
                    "cartpole",
                    "fixed_reward",
                    "sample_efficiency",
                    cfg.algorithm,
                    variant,
                    f"seed={cfg.seed}",
                ],
                config={
                    **asdict(cfg),
                    "task": task_name,
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
            validation_steps=cfg.evaluation_steps,
            validate_before_training=True,
            validation_interval_transitions=cfg.evaluation_interval_transitions,
            # Break-even, "task solved", and near-optimal control: the last one
            # is what separates algorithms once they all clear the first two.
            return_per_step_thresholds=(0.0, 3.0 / 250.0, 3.9 / 250.0),
        ),
        device=cfg.device,
        log_dir=log_dir,
        logger=logger,
    )
    return BuiltCartpoleBenchmark(
        env=env,
        runner=runner,
        log_dir=log_dir,
        video_recorder=video_recorder,
        training_video_dir=training_video_dir,
    )


def record_and_promote_video(built: BuiltCartpoleBenchmark, video_length: int) -> None:
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
