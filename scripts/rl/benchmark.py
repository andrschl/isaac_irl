# ruff: noqa: E402
"""Run one fixed-reward sample-efficiency benchmark."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

env_path = REPO_ROOT / ".env"
if env_path.exists():
    from dotenv import load_dotenv

    load_dotenv(env_path)


def _add_override_arguments(parser: argparse.ArgumentParser) -> None:
    """Overrides that keep every other value from the task profile."""
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--transition_budget", type=int)
    parser.add_argument("--evaluation_interval_transitions", type=int)
    parser.add_argument("--evaluation_steps", type=int)
    parser.add_argument("--action_clip", type=float)
    parser.add_argument("--value_backend", choices=("direct", "successor"))
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--updates_per_transition", type=float)
    parser.add_argument("--initial_critic_updates", type=int)
    parser.add_argument("--critic_learning_rate", type=float)
    parser.add_argument("--actor_learning_rate", type=float)
    parser.add_argument("--rollout_steps", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--entropy_temperature", type=float)
    parser.add_argument("--pmd_beta", type=float)
    parser.add_argument("--pmd_candidate_target_kl", type=float)
    parser.add_argument("--pmd_actor_projection_kl", type=float)
    parser.add_argument("--pmd_projection_steps", type=int)
    parser.add_argument("--pmd_critic_updates_per_projection", type=int)
    parser.add_argument("--pmd_candidate_actions", type=int)
    parser.add_argument("--pmd_actor_batch_size", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare fixed-reward RL algorithms by simulator samples."
    )
    parser.add_argument("--algorithm", required=True, choices=("ppo", "sac", "pmd"))
    parser.add_argument("--task", default="cartpole", choices=("cartpole", "lift"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--project", default="isaac-irl-sample-efficiency")
    parser.add_argument("--group")
    parser.add_argument("--run_name")
    parser.add_argument("--log_root", default="logs/benchmark")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_length", type=int, default=250)
    _add_override_arguments(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.video:
        args.enable_cameras = True
    simulation_app = AppLauncher(args).app

    from adapters.rsl_rl_benchmark_factory import (
        build_fixed_reward_benchmark,
        record_and_promote_video,
    )

    built = None
    exit_code = 0
    try:
        from config.benchmark_cli import build_benchmark_cfg

        cfg = build_benchmark_cfg(args)
        built = build_fixed_reward_benchmark(cfg)
        built.runner.learn_to_transition_budget(cfg.task.transition_budget)
        if cfg.record_video:
            record_and_promote_video(built, cfg.video_length)
    except BaseException:
        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        if built is not None:
            built.runner.finish_logger(exit_code=exit_code)
            built.env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
