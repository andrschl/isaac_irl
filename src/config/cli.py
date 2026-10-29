"""The single command-line surface for training.

Flags exist for the settings that get typed at a prompt. Everything else in the
schema is reachable with ``--set section.key=value``, so the CLI never has to
grow a twin of the YAML.
"""

from __future__ import annotations

import argparse
from typing import Any

_SET_HELP = (
    "Override any config value by dotted path, e.g. "
    "--set algo.train.critic_updates_per_batch=1000 --set runner.save_interval=10. "
    "Repeatable."
)


def build_arg_parser(app_launcher_cls: Any | None = None) -> argparse.ArgumentParser:
    """Build the training parser, with Isaac's launcher flags when available."""
    parser = argparse.ArgumentParser(
        description="Train a policy with IRL, RL, and optional BC regularization."
    )

    experiment = parser.add_argument_group("experiment")
    experiment.add_argument("--task", type=str, default=None, help="Isaac task name.")
    experiment.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment YAML. Defaults to configs/<task-slug>/experiment.yaml.",
    )
    experiment.add_argument("--seed", type=int, default=None)
    experiment.add_argument("--num_envs", type=int, default=None)
    experiment.add_argument("--max_iterations", type=int, default=None)
    experiment.add_argument(
        "--transition_budget",
        type=int,
        default=None,
        help=(
            "Stop after this many retained training transitions (may overshoot by a batch). "
            "For an all-interaction cap use --set runner.total_transition_budget=N."
        ),
    )
    if app_launcher_cls is None:
        # Isaac's AppLauncher owns --device when present. Keep the local flag
        # only for pure-Python callers that build this parser without Isaac.
        experiment.add_argument("--device", type=str, default=None)
    experiment.add_argument("--experiment_name", type=str, default=None)
    experiment.add_argument("--run_name", type=str, default=None)

    algorithm = parser.add_argument_group("algorithm")
    algorithm.add_argument(
        "--algorithm", type=str, default=None, choices=("ppo", "pmd")
    )
    algorithm.add_argument(
        "--value_backend",
        type=str,
        default=None,
        choices=("direct", "successor"),
        help="Action-value representation for PMD.",
    )
    algorithm.add_argument("--learning_rate", type=float, default=None)

    reward = parser.add_argument_group("reward")
    reward.add_argument(
        "--reward_source",
        type=str,
        default=None,
        choices=("learned", "environment", "environment_plus_learned"),
        help="Maximize the learned reward, or the environment's own.",
    )
    reward.add_argument(
        "--fit_diagnostic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="With --reward_source environment, still fit the reward model.",
    )
    reward.add_argument("--expert_data_path", type=str, default=None)
    reward.add_argument("--expert_num_trajectories", type=int, default=None)
    reward.add_argument("--bc_weight", type=float, default=None)

    run = parser.add_argument_group("run")
    run.add_argument("--logger", type=str, default=None, choices=("wandb", "noop"))
    run.add_argument("--log_project_name", type=str, default=None)
    run.add_argument("--validation_interval", type=int, default=None)
    run.add_argument("--resume", action="store_true", default=None)
    run.add_argument("--load_run", type=str, default=None)
    run.add_argument("--checkpoint", type=str, default=None)
    run.add_argument(
        "--init_policy_checkpoint",
        type=str,
        default=None,
        help="Warm-start the actor from a checkpoint without resuming a run.",
    )

    video = parser.add_argument_group("video")
    video.add_argument("--video", action="store_true", default=False)
    video.add_argument("--video_length", type=int, default=250)
    video.add_argument("--video_interval_iterations", type=int, default=None)
    video.add_argument(
        "--video_final_only",
        action="store_true",
        default=False,
        help="Record only the post-training clips.",
    )

    parser.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help=_SET_HELP
    )

    if app_launcher_cls is not None:
        app_launcher_cls.add_app_launcher_args(parser)
        if not any(action.dest == "disable_fabric" for action in parser._actions):
            run.add_argument(
                "--disable_fabric",
                action="store_true",
                default=False,
                help="Disable Isaac's Fabric scene updates.",
            )
    return parser


def build_checkpoint_evaluation_parser(
    app_launcher_cls: Any | None = None,
) -> argparse.ArgumentParser:
    """Standalone finite-horizon successor-PMD checkpoint diagnostics."""
    parser = argparse.ArgumentParser(
        description="Evaluate saved successor PMD policies without training."
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--bc_coverage",
        action="store_true",
        help="Compare repeated stochastic rollouts from sampled BC members with one BC baseline on paired starts.",
    )
    parser.add_argument(
        "--baseline_checkpoint",
        default=None,
        help="Single BC actor used for repeated-rollout coverage comparison.",
    )
    parser.add_argument(
        "--coverage_trials",
        type=int,
        default=8,
        help="Independent full trajectories per initial state and condition.",
    )
    parser.add_argument(
        "--coverage_member_sampling",
        choices=("with_replacement", "without_replacement"),
        default="with_replacement",
        help="How ensemble members are assigned to repeated trajectories from each initial state.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expert_data_path", default=None)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--episodes_per_seed", type=int, default=32)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("deterministic", "stochastic"),
        default=["deterministic", "stochastic"],
        help="Evaluate only the requested policy modes; each rollout costs interactions.",
    )
    parser.add_argument("--evaluation_seeds", nargs="+", type=int, default=[1001, 1002])
    parser.add_argument("--logger", choices=("noop", "wandb"), default="wandb")
    parser.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help=_SET_HELP
    )
    parser.set_defaults(video=False)
    if app_launcher_cls is not None:
        app_launcher_cls.add_app_launcher_args(parser)
    else:
        parser.add_argument("--device", default=None)
    if not any(action.dest == "disable_fabric" for action in parser._actions):
        parser.add_argument("--disable_fabric", action="store_true", default=False)
    return parser


def build_record_parser(app_launcher_cls: Any | None = None) -> argparse.ArgumentParser:
    """Build the demonstration-recording parser.

    Recording is not a second experiment definition: the task, its variant, its
    reward terms and the action bound all come from the experiment config the
    demonstrations are being recorded *for*, so what is left on the command
    line is the teacher and where to put the file.
    """
    parser = argparse.ArgumentParser(
        description="Record demonstrations by rolling out a trained policy."
    )
    parser.add_argument("--task", type=str, required=True, help="Isaac task name.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Experiment YAML defining the task these demonstrations are for. "
            "Defaults to configs/<task-slug>/experiment.yaml."
        ),
    )
    parser.add_argument(
        "--teacher_config",
        type=str,
        default=None,
        help=(
            "Experiment YAML whose actor architecture the checkpoint matches. "
            "Defaults to --config; give it when the teacher was trained on a "
            "different observation view than the task being recorded."
        ),
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Teacher policy to roll out."
    )
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument(
        "--max_attempted_episodes",
        type=int,
        default=None,
        help="Optional recording budget, including failures; stop before a complete vector wave would exceed it.",
    )
    parser.add_argument(
        "--demo_length",
        type=int,
        default=None,
        help="Assert the task's episode length; recording always keeps whole episodes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where demos.hdf5 is written. Defaults to logs/demos/<task-slug>/.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Replace an existing demonstration file instead of refusing to.",
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video_length", type=int, default=250)
    parser.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help=_SET_HELP
    )
    if app_launcher_cls is not None:
        app_launcher_cls.add_app_launcher_args(parser)
        if not any(action.dest == "disable_fabric" for action in parser._actions):
            parser.add_argument("--disable_fabric", action="store_true", default=False)
    else:
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--disable_fabric", action="store_true", default=False)
    return parser


def build_behavior_cloning_parser() -> argparse.ArgumentParser:
    """Build the offline behavioral-cloning parser.

    Offline BC touches no simulator, so it takes no Isaac launcher flags. It
    reads the same experiment config as the online arms, because the actor it
    writes has to be the actor they start from.
    """
    parser = argparse.ArgumentParser(
        description="Fit an actor to demonstrations offline, with no simulator."
    )
    parser.add_argument("--task", type=str, default=None, help="Isaac task name.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment YAML. Defaults to configs/<task-slug>/experiment.yaml.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--expert_data_path", type=str, default=None)
    parser.add_argument("--expert_num_trajectories", type=int, default=None)
    parser.add_argument(
        "--init_policy_checkpoint",
        type=str,
        default=None,
        help="Refine a saved actor with a fresh BC optimizer instead of initializing randomly.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=10_000,
        help="Adam steps over the demonstration set.",
    )
    parser.add_argument("--loss_type", choices=("mse", "nll"), default="mse")
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--final_learning_rate",
        type=float,
        default=None,
        help="Optional cosine-decay endpoint; the entire fit budget is used.",
    )
    parser.add_argument(
        "--l2_weight",
        type=float,
        default=0.0,
        help="Coefficient of half the squared mean-policy matrix norm; biases and exploration are excluded.",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.0,
        help="Whole-episode validation split for diagnostics; use 0 for the final all-demo fit.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--log_interval", type=int, default=500, help="Steps between loss reports."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Checkpoint path the fitted actor is written to.",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help=_SET_HELP
    )
    return parser


def build_bc_regularization_parser() -> argparse.ArgumentParser:
    """Whole-episode cross-validation followed by a fit on all demonstrations."""
    parser = build_behavior_cloning_parser()
    parser.description = (
        "Select BC L2 by episode-fold validation, then refit all demos."
    )
    parser.add_argument(
        "--l2_candidates", nargs="+", type=float, default=[0.0, 1e-6, 1e-5]
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.set_defaults(
        gradient_steps=40_000, final_learning_rate=3e-6, log_interval=2000
    )
    return parser


def build_spread_calibration_parser() -> argparse.ArgumentParser:
    """Fit only a smooth std head around saved complete MSE BC means."""
    parser = argparse.ArgumentParser(description="Calibrate BC spread on ten demos.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expert_data_path", required=True)
    parser.add_argument("--mean_checkpoint", required=True)
    parser.add_argument("--fold_checkpoint_dir", required=True)
    parser.add_argument(
        "--std_floors", nargs="+", type=float, default=[0.001, 0.005, 0.02]
    )
    parser.add_argument("--gradient_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--final_learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help=_SET_HELP
    )
    return parser


def parse_overrides(assignments: list[str]) -> dict[str, Any]:
    """Turn ``["a.b=1", "c=true"]`` into a nested mapping of parsed values."""
    overrides: dict[str, Any] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(
                f"--set expects PATH=VALUE, got {assignment!r} (no '=' found)."
            )
        path, _, raw_value = assignment.partition("=")
        node = overrides
        keys = path.strip().split(".")
        if not all(keys):
            raise ValueError(f"--set path must not have empty segments: {path!r}.")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(
                    f"--set path {path!r} conflicts with an earlier value."
                )
        node[keys[-1]] = _parse_value(raw_value)
    return overrides


def _parse_value(raw: str) -> Any:
    """Parse a YAML-ish scalar: null, bool, number, list, or string."""
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_parse_value(item) for item in inner.split(",")]
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overrides`` into ``base``, returning a new mapping."""
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = merge_overrides(current, value)
        else:
            merged[key] = value
    return merged


def build_evaluation_dashboard_parser() -> argparse.ArgumentParser:
    """Publish saved checkpoint evaluations; no simulator or training flags."""
    parser = argparse.ArgumentParser(
        description="Compare checkpoint evaluations on environment-interaction axes."
    )
    parser.add_argument("status_file", help="Comparison controller's status.json.")
    parser.add_argument("--entity", required=True, help="W&B entity.")
    parser.add_argument("--project", required=True, help="W&B project.")
    parser.add_argument(
        "--watch", action="store_true", help="Follow until the controller stops."
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def build_sequential_sweep_parser() -> argparse.ArgumentParser:
    """Launch the bounded Lift PMD sweep with one Isaac process at a time."""
    parser = argparse.ArgumentParser(
        description="Sequential successor-PMD tuning on easy Lift."
    )
    parser.add_argument(
        "output_dir", help="Fresh artifact directory under logs/benchmarks."
    )
    parser.add_argument(
        "--base-config", default="configs/lift_cube_franka/pmd_easy.yaml"
    )
    parser.add_argument(
        "--reference-status", required=True, help="Previous PPO comparison status.json."
    )
    parser.add_argument(
        "--evaluator", required=True, help="Previously verified checkpoint evaluator."
    )
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--entity", default="andrschl")
    parser.add_argument("--project", default="isaac_irl")
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def build_parallel_sweep_parser() -> argparse.ArgumentParser:
    """Run the two screening finalists concurrently, at a longer sample budget."""
    parser = argparse.ArgumentParser(description="Two-learner successor-PMD follow-up.")
    parser.add_argument("output_dir", help="Fresh directory under logs/benchmarks.")
    parser.add_argument(
        "--reference-status", required=True, help="Completed screening status.json."
    )
    parser.add_argument("--training-budget", type=int, default=3200000)
    parser.add_argument("--max-hours", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--entity", default="andrschl")
    parser.add_argument("--project", default="isaac_irl")
    parser.add_argument("--prepare-only", action="store_true")
    return parser
