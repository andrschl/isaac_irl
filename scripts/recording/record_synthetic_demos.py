"""Collect synthetic demonstrations from a trained RSL-RL policy."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

from isaaclab.app import AppLauncher

# local imports
import cli_args as cli_args  # isort: skip


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect synthetic demos with an RSL-RL policy.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Task name.")
    parser.add_argument("--num_demos", type=int, default=1000, help="Number of episodes to collect.")
    parser.add_argument(
        "--demo_length",
        type=int,
        default=None,
        help="Maximum episode length override. Defaults to the env max episode length.",
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during collection.")
    parser.add_argument(
        "--video_length",
        type=int,
        default=None,
        help="Recorded clip length in steps. Defaults to the effective demo length.",
    )
    parser.add_argument("--feature_key", type=str, default="features", help="Dataset key used for features.")
    parser.add_argument(
        "--feature_type",
        type=str,
        default="manager",
        choices=("manager", "custom"),
        help="Reward feature source. 'custom' is reserved and currently unsupported.",
    )
    parser.add_argument(
        "--ignored_reward_terms",
        type=str,
        default="",
        help="Comma-separated manager reward terms to ignore.",
    )
    parser.add_argument(
        "--status_interval_steps",
        type=int,
        default=10,
        help="How often to print status logs (environment steps).",
    )
    parser.add_argument(
        "--demo_log_interval",
        type=int,
        default=10,
        help="How often to print collected demo count.",
    )

    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


parser = _build_arg_parser()
args_cli = parser.parse_args()
if args_cli.feature_type == "custom":
    raise NotImplementedError(
        "--feature_type custom is not implemented yet. Use --feature_type manager."
    )
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

try:
    from collectors import RobomimicDataCollector
    from reward_features.manager_based import manager_based_reward_feature_dict
except ModuleNotFoundError:
    repo_src = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(repo_src))
    from collectors import RobomimicDataCollector
    from reward_features.manager_based import manager_based_reward_feature_dict


def _is_mapping_like(value: Any) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def _as_obs_mapping(obs: Any) -> Any:
    """Return mapping-like observations suitable for dataset storage."""
    if _is_mapping_like(obs):
        return obs
    if isinstance(obs, torch.Tensor):
        return {"policy": obs}
    raise TypeError(f"Expected observations to be mapping-like or tensor, got {type(obs)!r}.")


def _try_set_demo_length(env: Any, demo_length: int) -> None:
    for target in (env, getattr(env, "unwrapped", None)):
        if target is None:
            continue
        if hasattr(target, "max_episode_length"):
            try:
                target.max_episode_length = int(demo_length)
                return
            except Exception:
                pass


def _get_env_max_episode_length(env: Any) -> int | None:
    for target in (env, getattr(env, "unwrapped", None)):
        if target is None:
            continue
        value = getattr(target, "max_episode_length", None)
        if value is None:
            continue
        try:
            max_length = int(value)
        except Exception:
            continue
        if max_length > 0:
            return max_length
    return None


def _resolve_demo_length(env: Any, demo_length_override: int | None) -> int:
    if demo_length_override is not None:
        demo_length = max(1, int(demo_length_override))
        _try_set_demo_length(env, demo_length)
        return demo_length

    env_demo_length = _get_env_max_episode_length(env)
    if env_demo_length is not None:
        return env_demo_length

    raise ValueError(
        "Could not infer env max episode length. "
        "Pass --demo_length explicitly."
    )


def _step_env(env: Any, actions: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor]:
    """Normalize env.step output to (obs, rewards[N], dones[N])."""
    out = env.step(actions)
    if not isinstance(out, tuple):
        raise TypeError(f"Expected tuple from env.step(...), got {type(out)!r}.")

    if len(out) == 4:
        obs, rewards, dones, _ = out
    elif len(out) == 5:
        obs, rewards, terminated, truncated, _ = out
        dones = torch.as_tensor(terminated) | torch.as_tensor(truncated)
    else:
        raise ValueError(f"Expected env.step(...) to return 4 or 5 values, got {len(out)}.")

    rewards_tensor = rewards if isinstance(rewards, torch.Tensor) else torch.as_tensor(rewards)
    dones_tensor = dones if isinstance(dones, torch.Tensor) else torch.as_tensor(dones)
    return obs, rewards_tensor, dones_tensor.to(torch.bool)


def _get_policy_observations(env: Any) -> Any:
    """Return current policy observations from the vectorized env wrapper."""
    if hasattr(env, "get_observations"):
        out = env.get_observations()
    else:
        out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def main() -> None:
    if args_cli.task is None:
        raise ValueError("--task is required.")
    if args_cli.feature_key != "features":
        raise ValueError(
            f"--feature_key must be 'features' for IRL compatibility, got '{args_cli.feature_key}'."
        )

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    max_demo_length = _resolve_demo_length(env, args_cli.demo_length)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    demo_dir = os.path.abspath(os.path.join("logs", "demos", agent_cfg.experiment_name))
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    print(f"[INFO] Saving demos in directory: {demo_dir}")
    print(f"[INFO] Effective demo length: {max_demo_length} steps.")

    if args_cli.video:
        video_length = int(args_cli.video_length) if args_cli.video_length is not None else max_demo_length
        video_length = max(1, video_length)
        video_kwargs = {
            "video_folder": os.path.join(demo_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "name_prefix": "collection_episode",
            "disable_logger": True,
        }
        print("[INFO] Recording one collection clip.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env)

    print("[INFO] Initializing RSL-RL runner...")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print("[INFO] Loading checkpoint...")
    runner.load(resume_path)
    print("[INFO] Checkpoint loaded.")

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    collector = RobomimicDataCollector(
        task_name=args_cli.task,
        output_dir=demo_dir,
        filename="demos.hdf5",
        num_demos=int(args_cli.num_demos),
    )
    collector.reset()

    ignored_terms = {term.strip() for term in args_cli.ignored_reward_terms.split(",") if term.strip()}

    policy_obs = _get_policy_observations(env)
    obs_map = _as_obs_mapping(policy_obs)
    episode_steps = torch.zeros(int(env.num_envs), dtype=torch.long, device=env.device)

    collected_episodes = 0
    loop_steps = 0

    try:
        with torch.inference_mode():
            while not collector.is_stopped():
                collector.add("obs", obs_map)

                named_features = manager_based_reward_feature_dict(
                    env=env,
                    ignored_reward_terms=ignored_terms,
                    device=env.unwrapped.device,
                )
                collector.add(args_cli.feature_key, named_features)

                actions = policy(policy_obs)
                collector.add("actions", actions)

                policy_obs, rewards, dones = _step_env(env, actions)
                obs_map = _as_obs_mapping(policy_obs)
                env_dones = dones.reshape(-1).to(torch.bool)
                episode_steps += 1
                timeout_dones = episode_steps >= max_demo_length
                effective_dones = env_dones | timeout_dones
                collector.add("rewards", rewards)
                collector.add("dones", effective_dones)

                done_env_ids = effective_dones.nonzero(as_tuple=False).squeeze(-1)
                collector.flush(done_env_ids)
                if done_env_ids.numel() > 0:
                    episode_steps[done_env_ids] = 0

                if done_env_ids.numel() > 0:
                    collected_episodes += int(done_env_ids.numel())
                    if collected_episodes % max(1, int(args_cli.demo_log_interval)) == 0:
                        print(f"[INFO] Collected {collected_episodes}/{args_cli.num_demos} demos.")

                loop_steps += 1
                if loop_steps % max(1, int(args_cli.status_interval_steps)) == 0:
                    print(
                        f"[INFO] Recorder heartbeat: step={loop_steps}, "
                        f"collected={collected_episodes}/{args_cli.num_demos}"
                    )
    finally:
        collector.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
