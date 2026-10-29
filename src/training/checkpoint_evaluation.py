"""Evaluate saved finite-horizon PMD checkpoints, never updating or filling replay."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from adapters.pmd import PmdAdapter
from algorithms.off_policy.values.backends import SuccessorActionValue
from algorithms.off_policy.values.diagnostics import successor_value_diagnostics
from config import load_experiment_cfg
from config.cli import build_checkpoint_evaluation_parser
from reward.episode_success import episode_success
from training import build
from training.bc_coverage import (
    aggregate_coverage,
    evaluate_bc_coverage,
    publish_bc_coverage,
)
from training.pipeline import (
    REPO_ROOT,
    _app_launcher_cls,
    isaac_lifecycle,
    load_runtime_deps,
)


def evaluate_checkpoint(
    algorithm: PmdAdapter,
    *,
    env,
    episodes: int,
    seeds: list[int],
    diagnose_values: bool = True,
    modes: tuple[str, ...] = ("deterministic", "stochastic"),
) -> list[dict]:
    """Collect deterministic/stochastic evaluations with isolated action RNGs.

    Returns scalar result rows. Only stochastic episodes diagnose Q^pi; the
    deterministic policy is different and is reported as task performance only.
    """
    value = algorithm.learner.value
    if diagnose_values and not isinstance(value, SuccessorActionValue):
        raise ValueError("Checkpoint diagnostics require the successor backend.")
    if algorithm.replay.transition_count or algorithm.replay.demonstration_count:
        raise ValueError("Standalone evaluation must start with empty replay.")
    algorithm.eval_mode()
    names = algorithm.collector.steps.feature_tap.spec.names
    rows = []
    if (
        not modes
        or len(set(modes)) != len(modes)
        or set(modes) - {"deterministic", "stochastic"}
    ):
        raise ValueError("Require distinct deterministic/stochastic evaluation modes.")
    for mode in modes:
        for seed in seeds:
            env.seed(seed)
            generator = torch.Generator(
                device=algorithm.collector.steps.device
            ).manual_seed(seed + 100_003)

            def stochastic_actions(observations):
                return algorithm.actor.sample_with_log_probability(
                    observations, generator=generator
                )[0]

            batch = algorithm.collector.collect(
                policy_version=algorithm.learner.policy_version,
                episodes=episodes,
                action_fn=algorithm.actor.act_inference
                if mode == "deterministic"
                else stochastic_actions,
            )
            transitions = batch.transitions
            success = episode_success(
                transitions.features,
                names,
                batch.mask,
                terminated=transitions.terminated,
                truncated=transitions.truncated,
            )
            if success is None:
                raise ValueError("Evaluation task does not expose a success feature.")
            row = {
                "mode": mode,
                "seed": seed,
                "episodes": batch.episode_count,
                "successes": int(success.sum()),
                "return_mean": float(batch.returns.mean()),
                "retained_transitions": batch.transition_count,
                "environment_transitions": batch.environment_transitions,
                "discarded_transitions": batch.environment_transitions
                - batch.transition_count,
                "terminated_episodes": int(
                    (transitions.terminated & batch.mask).any(1).sum()
                ),
                "feature_means": {
                    name: float(transitions.features[:, :, column][batch.mask].mean())
                    for column, name in enumerate(names)
                },
            }
            if "lifting_object" in names:
                lifted = transitions.features[:, :, names.index("lifting_object")] > 0.5
                row["lifted_episodes"] = int((lifted & batch.mask).any(1).sum())
            if mode == "stochastic" and diagnose_values:
                row["values"] = successor_value_diagnostics(
                    value,
                    algorithm.actor,
                    batch,
                    entropy_rate=algorithm.learner.entropy_rate,
                    feature_names=names,
                )
            rows.append(row)
            print(
                f"[INFO] {mode} seed={seed}: {row['successes']}/{episodes} successes, "
                f"return={row['return_mean']:.4f}, physical_interactions={batch.environment_transitions}",
                flush=True,
            )
    if algorithm.replay.transition_count or algorithm.replay.demonstration_count:
        raise RuntimeError("Evaluation inserted data into replay.")
    return rows


def load_evaluation_checkpoint(
    algorithm,
    reward_model,
    checkpoint: dict,
    *,
    reward_feature_names: tuple[str, ...] | None = None,
) -> dict:
    """Load a trained PMD stack or a BC actor; never treat random critics as fitted."""
    if "actor_state_dict" in checkpoint:
        if "behavior_cloning" not in checkpoint:
            raise ValueError("Actor-only evaluation requires BC checkpoint metadata.")
        algorithm.actor.model.load_state_dict(
            checkpoint["actor_state_dict"], strict=True
        )
        algorithm.learner.previous_policy = algorithm.actor.snapshot()
        algorithm.learner.policy_version = 0
        return {
            "has_value_estimates": False,
            "behavior_cloning": checkpoint["behavior_cloning"],
            "training_environment_transitions": 0,
            "training_retained_transitions": 0,
            "training_episodes": 0,
        }
    if reward_model is not None:
        reward_state = checkpoint["reward_learner_state"]
        if (
            reward_feature_names is not None
            and "feature_names" in reward_state
            and reward_state["feature_names"] != list(reward_feature_names)
        ):
            raise ValueError(
                "Evaluation learned-feature names do not match the checkpoint."
            )
        if "feature_scales" in reward_state and (
            not hasattr(reward_model, "feature_scales")
            or not torch.equal(
                torch.as_tensor(reward_state["feature_scales"], device="cpu"),
                reward_model.feature_scales.cpu(),
            )
        ):
            raise ValueError("Evaluation feature scales do not match the checkpoint.")
    algorithm.learner.load_state(
        checkpoint["rl_alg_state"]["learner"], load_optimizer=False
    )
    if reward_model is not None:
        reward_model.load_state_dict(
            checkpoint["reward_learner_state"]["reward_model_state_dict"], strict=True
        )
    counts = checkpoint["interaction_counts"]
    return {
        "has_value_estimates": True,
        "training_environment_transitions": counts["environment_transitions"]
        - counts["evaluation_transitions"]
        - counts["imitator_transitions"],
        "training_retained_transitions": checkpoint["global_timestep"],
        "training_episodes": counts["training_episodes"],
    }


def load_bc_members(
    actor_template, checkpoints: list[Path], *, device: str
) -> tuple[list, list[dict]]:
    """Load independently fitted BC members with one actor/data contract."""
    if len(checkpoints) < 2 or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("BC ensemble requires at least two distinct checkpoints.")
    members = []
    member_facts = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if "actor_state_dict" not in checkpoint or "behavior_cloning" not in checkpoint:
            raise ValueError(
                f"BC ensemble member is not an actor-only BC checkpoint: {path}."
            )
        facts = checkpoint["behavior_cloning"]
        if "policy" not in facts or "expert_data_path" not in facts:
            raise ValueError(
                f"BC ensemble member lacks policy/data provenance: {path}."
            )
        if member_facts and (
            facts["policy"] != member_facts[0]["policy"]
            or Path(facts["expert_data_path"]).resolve()
            != Path(member_facts[0]["expert_data_path"]).resolve()
            or facts.get("expert_pairs") != member_facts[0].get("expert_pairs")
        ):
            raise ValueError(
                "BC ensemble members must share policy, demonstrations, and pair count."
            )
        member = actor_template.snapshot()
        member.model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        members.append(member)
        member_facts.append(facts)
    return members, member_facts


def evaluation_metrics(
    result: dict, *, cumulative_evaluation_interactions: int
) -> dict[str, float]:
    """Aggregate by attempted episodes / valid transitions; explicit physical axes."""
    training = result["training_environment_transitions"]
    metrics = {
        "Samples/training_environment_interactions": training,
        "Samples/training_trajectories": result["training_episodes"],
        "Samples/evaluation_interactions": cumulative_evaluation_interactions,
        "Samples/total_environment_interactions": training
        + cumulative_evaluation_interactions,
    }
    for mode in ("deterministic", "stochastic"):
        rows = [row for row in result["rows"] if row["mode"] == mode]
        if not rows:
            continue
        episodes = sum(row["episodes"] for row in rows)
        returns = sum(row["episodes"] * row["return_mean"] for row in rows)
        retained = sum(row["retained_transitions"] for row in rows)
        prefix = f"Evaluation/{mode}"
        metrics[f"{prefix}/success_rate"] = (
            sum(row["successes"] for row in rows) / episodes
        )
        metrics[f"{prefix}/episode_return_mean"] = returns / episodes
        metrics[f"{prefix}/reward_per_step"] = returns / retained
        metrics[f"{prefix}/episodes"] = episodes
        metrics[f"{prefix}/failure_rate"] = (
            sum(row["terminated_episodes"] for row in rows) / episodes
        )
        if "lifted_episodes" in rows[0]:
            metrics[f"{prefix}/lift_rate"] = (
                sum(row["lifted_episodes"] for row in rows) / episodes
            )
        for name in rows[0].get("feature_means", {}):
            metrics[f"{prefix}/features/{name}"] = (
                sum(
                    row["feature_means"][name] * row["retained_transitions"]
                    for row in rows
                )
                / retained
            )
        if mode == "stochastic" and "values" in rows[0]:
            for name in rows[0]["values"]:
                # Means across reset seeds, explicitly labelled; RMSE is not
                # presented as an exactly pooled conditional-value error.
                metrics[f"Values/seed_mean/{name}"] = sum(
                    row["values"][name] for row in rows
                ) / len(rows)
    return metrics


def publish_results(
    results: list[dict], *, cfg, directory: Path, run_name: str | None = None
) -> None:
    """A separate W&B evaluation stream; never resume or rewrite the learner run."""
    import wandb

    with wandb.init(
        project=cfg.wandb_project,
        name=run_name or directory.name,
        group=cfg.experiment_name + "_evaluations",
        job_type="checkpoint-evaluation",
        dir=str(directory),
        tags=["evaluation", "batch-irl", "value-diagnostics"],
        config={
            "training_seed": cfg.seed,
            "reward": asdict(cfg.reward),
            "bc/weight": cfg.bc.weight,
            "source_checkpoints": [r["checkpoint"] for r in results],
            "behavior_cloning": results[0].get("behavior_cloning"),
            "evaluation_protocol": results[0]["protocol"],
            "x_axis": "physical training interactions",
            "value_error": "sampled stochastic returns, not exact conditional Q",
        },
    ) as run:
        run.define_metric("Samples/*", hidden=True)
        for prefix in ("Evaluation/*", "Values/*"):
            run.define_metric(
                prefix,
                step_metric="Samples/training_environment_interactions",
                step_sync=False,
            )
        cumulative = 0
        for result in results:
            cumulative += sum(row["environment_transitions"] for row in result["rows"])
            run.log(
                evaluation_metrics(
                    result, cumulative_evaluation_interactions=cumulative
                )
            )
        print(f"[INFO] Evaluation W&B: {run.url}", flush=True)


def main(argv: list[str] | None = None) -> None:
    launcher = _app_launcher_cls()
    args = build_checkpoint_evaluation_parser(launcher).parse_args(argv)
    if args.episodes_per_seed <= 0 or len(set(args.evaluation_seeds)) != len(
        args.evaluation_seeds
    ):
        raise ValueError(
            "Require positive episodes_per_seed and distinct evaluation seeds."
        )
    if args.bc_coverage and (
        args.baseline_checkpoint is None
        or args.coverage_trials <= 0
        or args.modes != ["stochastic"]
    ):
        raise ValueError(
            "BC coverage requires --baseline_checkpoint, positive "
            "--coverage_trials, and --modes stochastic."
        )
    cfg, task = load_experiment_cfg(args=args, config_root=REPO_ROOT / "configs")
    if (
        cfg.is_on_policy
        or cfg.algorithm.value.type != "successor"
        or cfg.gamma != 1.0
        or not cfg.env.is_finite_horizon
    ):
        raise ValueError(
            "This evaluator requires finite-horizon successor PMD with gamma=1."
        )
    # Construct exactly the task and networks, but no BC, demo replay, warm-up,
    # reward optimizer, runner, or training loop. Load trained weights below.
    replay_cfg = replace(cfg.algorithm.train.replay, demonstration_path=None)
    cfg = replace(
        cfg,
        init_policy_checkpoint=None,
        algorithm=replace(
            cfg.algorithm, train=replace(cfg.algorithm.train, replay=replay_cfg)
        ),
    )
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    checkpoints = [Path(path).resolve(strict=True) for path in args.checkpoints]
    if len({path.stem for path in checkpoints}) != len(checkpoints):
        raise ValueError(
            "Checkpoint filenames must be distinct within one evaluation output directory."
        )
    app = launcher(args).app
    with isaac_lifecycle(app, args) as state:
        deps = load_runtime_deps(launcher)
        env_cfg = build.build_env_cfg(
            cfg, parse_env_cfg=deps.parse_env_cfg, use_fabric=not args.disable_fabric
        )
        state.env = deps.gym.make(task, cfg=env_cfg)
        env = build.wrap_env(
            state.env, RslRlVecEnvWrapper=deps.RslRlVecEnvWrapper, seed=cfg.seed
        )
        state.env = env
        observations = env.get_observations().to(cfg.device)
        feature_source = build.build_feature_source(cfg, env=env)
        reward_model = build.build_reward_model(
            cfg, env=env, feature_source=feature_source, device=cfg.device
        )
        scorer = build.build_reward_scorer(
            cfg, env=env, feature_source=feature_source, reward_model=reward_model
        )
        algorithm = build.build_algorithm(
            cfg,
            env=env,
            observations=observations,
            obs_groups={},
            feature_source=feature_source,
            reward=scorer,
            bc_alg=None,
            deps=deps,
        )
        if not isinstance(algorithm, PmdAdapter):
            raise TypeError("Expected a PMD adapter.")
        if args.bc_coverage:
            members, member_facts = load_bc_members(
                algorithm.actor, checkpoints, device=cfg.device
            )
            baseline_path = Path(args.baseline_checkpoint).resolve(strict=True)
            baseline_state = torch.load(
                baseline_path, map_location=cfg.device, weights_only=False
            )
            if (
                "actor_state_dict" not in baseline_state
                or "behavior_cloning" not in baseline_state
            ):
                raise ValueError(
                    "BC coverage baseline must be an actor-only BC checkpoint."
                )
            baseline = algorithm.actor.snapshot()
            baseline.model.load_state_dict(
                baseline_state["actor_state_dict"], strict=True
            )
            print(
                f"[INFO] Evaluating {args.coverage_trials} stochastic trajectories "
                f"per initial state from {len(checkpoints)} BC members versus one baseline",
                flush=True,
            )
            result = {
                "member_checkpoints": [str(path) for path in checkpoints],
                "baseline_checkpoint": str(baseline_path),
                "behavior_cloning": member_facts,
                "baseline_behavior_cloning": baseline_state["behavior_cloning"],
                "training_environment_transitions": 0,
                **evaluate_bc_coverage(
                    algorithm,
                    env=env,
                    baseline=baseline,
                    members=members,
                    episodes_per_seed=args.episodes_per_seed,
                    seeds=args.evaluation_seeds,
                    trials=args.coverage_trials,
                    member_sampling=args.coverage_member_sampling,
                ),
                "protocol": {
                    "env": asdict(cfg.env),
                    "policy": asdict(cfg.algorithm.policy),
                    "gamma": cfg.gamma,
                    "episodes_per_seed": args.episodes_per_seed,
                    "evaluation_seeds": args.evaluation_seeds,
                    "coverage_trials": args.coverage_trials,
                    "modes": ["stochastic"],
                    "member_assignment": (
                        "uniform with replacement"
                        if args.coverage_member_sampling == "with_replacement"
                        else "shuffled without-replacement cycles per initial state"
                    )
                    + ", one fixed BC member per entire episode",
                    "coverage_member_sampling": args.coverage_member_sampling,
                    "baseline": "independent action samples from one fixed BC policy",
                    "best_of_n": "fraction of matched initial states with at least one success in the first n sampled trajectories",
                    "paired_initial_observation_tolerance": 1e-5,
                },
            }
            result["summary"] = aggregate_coverage(
                result["rows"],
                additional_prefixes=(len(members),)
                if args.coverage_member_sampling == "without_replacement"
                and args.coverage_trials >= len(members)
                else (),
            )
            with (directory / "bc_coverage.json").open("x") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
            if args.logger == "wandb":
                publish_bc_coverage(result, cfg=cfg, directory=directory)
            return
        results = []
        for path in checkpoints:
            checkpoint = torch.load(path, map_location=cfg.device, weights_only=False)
            metadata = load_evaluation_checkpoint(
                algorithm,
                reward_model,
                checkpoint,
                reward_feature_names=build.learned_feature_spec(
                    cfg, feature_source
                ).names
                if reward_model is not None
                else None,
            )
            print(f"[INFO] Evaluating {path}", flush=True)
            result = {
                "checkpoint": str(path),
                **metadata,
                "rows": evaluate_checkpoint(
                    algorithm,
                    env=env,
                    episodes=args.episodes_per_seed,
                    seeds=args.evaluation_seeds,
                    diagnose_values=metadata["has_value_estimates"],
                    modes=tuple(args.modes),
                ),
                "protocol": {
                    "env": asdict(cfg.env),
                    "policy": asdict(cfg.algorithm.policy),
                    "gamma": cfg.gamma,
                    "episodes_per_seed": args.episodes_per_seed,
                    "evaluation_seeds": args.evaluation_seeds,
                    "modes": args.modes,
                    "value_error_note": "Stochastic sampled-return error includes return variance; final critic can lag the last policy projection.",
                },
            }
            with (directory / f"{path.stem}.json").open("x") as handle:
                json.dump(result, handle, indent=2, allow_nan=False)
            results.append(result)
        if args.logger == "wandb":
            # BC candidates all have x=0: separate streams identify different
            # fits, rather than implying they are successive RL checkpoints.
            bc_results = [
                result for result in results if not result["has_value_estimates"]
            ]
            for result in bc_results:
                publish_results(
                    [result],
                    cfg=cfg,
                    directory=directory,
                    run_name=f"{directory.name}/{Path(result['checkpoint']).stem}",
                )
            learned_results = [
                result for result in results if result["has_value_estimates"]
            ]
            if learned_results:
                publish_results(learned_results, cfg=cfg, directory=directory)
