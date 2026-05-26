"""argparse wiring for `scripts/irl/train_irl.py`.

Kept separate so the training script's `main()` is short. Matches the
pattern set by `scripts/rsl_rl/cli_args.py`.
"""

from __future__ import annotations

import argparse
from typing import Any


def build_arg_parser(app_launcher_cls: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an IRL agent with PPO + feature-buffer IRL.")

    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=1500, help="Length of recorded videos (steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between recordings (steps).")
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
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Environment / training seed.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
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

    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name for logging.")
    parser.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
    parser.add_argument("--resume", action="store_true", default=None, help="Resume from a previous checkpoint.")
    parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
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
