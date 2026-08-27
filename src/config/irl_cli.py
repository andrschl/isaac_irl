"""Pure argparse schema for IRL and fixed-reward robotics training."""

from __future__ import annotations

import argparse


def build_irl_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument(
        "--algorithm",
        choices={"ppo", "sac", "pmd"},
        default=None,
        help="Override algo.type from the experiment YAML.",
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
        "--bc_weight",
        type=float,
        default=None,
        help="BC mix: (1-w)·L_RL + w·L_BC. Zero disables BC data and gradients.",
    )
    parser.add_argument(
        "--bc_loss_type",
        type=str,
        choices={"nll", "mse"},
        default=None,
        help="BC loss type: nll (default) or mse.",
    )
    parser.add_argument(
        "--value_backend",
        choices={"direct", "successor"},
        default=None,
        help="For SAC/PMD, select scalar direct Q or reward-transferable successor Q.",
    )
    parser.add_argument(
        "--num_value_heads",
        type=int,
        default=None,
        help="For SAC/PMD, override the direct/successor ensemble head count.",
    )
    parser.add_argument(
        "--updates_per_transition",
        type=float,
        default=None,
        help="Off-policy critic UTD: optimizer steps per new simulator transition.",
    )
    parser.add_argument(
        "--initial_critic_updates",
        type=int,
        default=None,
        help="Off-policy critic-only updates after replay reaches learning_starts.",
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=None,
        help="Off-policy direct/successor critic optimizer learning rate.",
    )
    parser.add_argument(
        "--rollout_steps",
        type=int,
        default=None,
        help="Off-policy simulator steps collected per environment per cycle.",
    )
    parser.add_argument(
        "--pmd_beta",
        type=float,
        default=None,
        help="PMD old-policy exponent in pi_target proportional to pi_old^beta exp(cQ).",
    )
    parser.add_argument(
        "--pmd_critic_updates_per_projection",
        type=int,
        default=None,
        help="PMD critic steps evaluating the frozen policy before each projection.",
    )
    parser.add_argument(
        "--pmd_candidate_target_kl",
        type=float,
        default=None,
        help="PMD nonparametric target KL solved for by the Q-coefficient bisection.",
    )
    parser.add_argument(
        "--pmd_actor_projection_kl",
        type=float,
        default=None,
        help="PMD trust region: KL bound of the fitted actor from the frozen policy.",
    )
    parser.add_argument(
        "--entropy_temperature",
        type=float,
        default=None,
        help="SAC initial or PMD fixed Shannon-entropy coefficient.",
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
        "--use_ground_truth_reward_weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Initialize the linear reward model from the environment reward-manager "
        "weights and disable IRL reward updates.",
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
        "--init_policy_checkpoint",
        type=str,
        default=None,
        help="Initialize actor parameters only; mutually exclusive with --resume.",
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

    return parser
