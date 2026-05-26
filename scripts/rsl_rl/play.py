"""Script to evaluate a trained RSL-RL policy checkpoint."""

from __future__ import annotations

import argparse
import os

import torch
from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RL policy with RSL-RL.")
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record video during evaluation.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=1500,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument(
    "--num_steps",
    type=int,
    default=None,
    help="Stop after this many environment steps. None = run until SimulationApp closes.",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="Print a heartbeat every N env steps so the user sees progress in headless mode.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import importlib.metadata as _importlib_metadata

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import isaaclab_tasks  # noqa: F401


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    # Migrate legacy `policy=RslRlPpoActorCriticCfg(...)` to the new-style
    # `actor`/`critic` blocks required by rsl-rl >= 4.0. Without this,
    # `PPO.construct_algorithm` raises `KeyError: 'class_name'`.
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, _importlib_metadata.version("rsl-rl-lib")
    )

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
    )
    log_dir = os.path.dirname(resume_path)

    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO] Loading policy checkpoint: {resume_path}")
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_dir, exist_ok=True)
    try:
        runner.export_policy_to_jit(path=export_dir, filename="policy.pt")
    except Exception as exc:
        print(f"[WARN] Failed to export JIT policy: {exc}")
    try:
        runner.export_policy_to_onnx(path=export_dir, filename="policy.onnx")
    except Exception as exc:
        print(f"[WARN] Failed to export ONNX policy: {exc}")

    obs = env.get_observations()

    steps = 0
    max_steps = int(args_cli.num_steps) if args_cli.num_steps is not None else None
    log_interval = max(1, int(args_cli.log_interval))
    print(
        f"[INFO] Starting inference loop (max_steps={max_steps}, "
        f"log every {log_interval} steps)."
    )
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        steps += 1
        if steps % log_interval == 0:
            print(f"[INFO] play step {steps}")

        if args_cli.video and steps >= int(args_cli.video_length):
            print(
                f"[INFO] Reached video length ({args_cli.video_length} steps). Stopping."
            )
            break
        if max_steps is not None and steps >= max_steps:
            print(f"[INFO] Reached --num_steps={max_steps}. Stopping.")
            break

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
