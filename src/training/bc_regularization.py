"""Select BC regularization on whole trajectories, then refit all demonstrations.

No simulator, random transition split, early checkpoint selection, or success
target is used here. All candidates receive the same complete optimizer budget.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from algorithms.bc import BC
from algorithms.off_policy.actor import policy_observation_tensor
from config import load_experiment_cfg
from config.cli import build_bc_regularization_parser
from config.schema import BCCfg
from data.expert_obs_actions import load_obs_action_episodes
from storage.expert_dataset import ExpertObsActionDataset
from training.behavior_cloning import (
    actor_fit_metrics,
    build_actor,
    fit_actor,
    save_actor,
)
from training.pipeline import REPO_ROOT


def episode_folds(
    episode_count: int, fold_count: int, *, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Disjoint train/validation episode indices; each episode held out once."""
    if not 2 <= fold_count <= episode_count:
        raise ValueError("Require 2 <= fold_count <= episode_count.")
    order = torch.randperm(episode_count, generator=torch.Generator().manual_seed(seed))
    folds = []
    for validation in torch.tensor_split(order, fold_count):
        validation_indices = validation.tolist()
        training_indices = [
            index for index in order.tolist() if index not in validation_indices
        ]
        folds.append((training_indices, validation_indices))
    return folds


def run(args: argparse.Namespace) -> dict:
    """Fit from scratch, select final held-out BC loss, then refit all demos."""
    cfg, _ = load_experiment_cfg(args=args, config_root=REPO_ROOT / "configs")
    if (
        cfg.init_policy_checkpoint is not None
        or cfg.expert_data.num_trajectories is not None
    ):
        raise ValueError("Cross-validation starts fresh and uses every supplied demo.")
    if args.validation_fraction != 0.0 or args.l2_weight != 0.0:
        raise ValueError("Use --folds and --l2_candidates for cross-validation.")
    candidates = sorted(set(args.l2_candidates))
    if not candidates or any(not math.isfinite(x) or x < 0 for x in candidates):
        raise ValueError("l2_candidates must be finite, nonnegative values.")
    episodes = load_obs_action_episodes(cfg.expert_data.path)
    folds = episode_folds(len(episodes), args.folds, seed=cfg.seed)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = output.parent / f"{output.stem}_validation"
    if output.exists():
        raise FileExistsError(f"Preserve the existing checkpoint: {output}")
    directory.mkdir(exist_ok=False)
    records = []
    selection = f"final whole-episode cross-validation {args.loss_type.upper()}"

    def fit(label: str, training: list[int], validation: list[int], l2: float):
        # Matched initialization and minibatch RNG across lambda within a fold.
        torch.manual_seed(cfg.seed)
        bc = BC(
            cfg=BCCfg(
                weight=1.0,
                loss_type=args.loss_type,
                batch_size=args.batch_size,
                val_fraction=0,
            ),
            storage=ExpertObsActionDataset([episodes[index] for index in training]),
            val_storage=ExpertObsActionDataset(
                [episodes[index] for index in validation]
            )
            if validation
            else None,
            device=cfg.device,
        )
        observations, actions = bc.storage.all_pairs(cfg.device)
        policy = build_actor(
            cfg,
            observation_dim=policy_observation_tensor(observations).shape[-1],
            action_dim=actions.shape[-1],
            device=cfg.device,
        )
        facts = {
            "seed": cfg.seed,
            "expert_data_path": cfg.expert_data.path,
            "training_episodes": training,
            "validation_episodes": validation,
            "expert_pairs": len(bc.storage),
            "gradient_steps": args.gradient_steps,
            "learning_rate": args.learning_rate,
            "final_learning_rate": args.final_learning_rate,
            "l2_weight": l2,
            "batch_size": args.batch_size,
            "loss_type": args.loss_type,
            "policy": asdict(cfg.algorithm.policy),
            "selection": selection,
        }
        logger = None
        if cfg.logger == "wandb":
            import wandb

            logger = wandb.init(
                project=cfg.wandb_project,
                group=directory.name,
                name=f"{directory.name}/{label}",
                job_type="offline_bc",
                config=facts,
            )
            logger.define_metric("BC/optimizer_steps")
            logger.define_metric("BC/*", step_metric="BC/optimizer_steps")
        trace = []

        def record(step, metrics):
            trace.append({"step": step, **metrics})
            if logger is not None:
                logger.log(
                    {
                        "BC/optimizer_steps": step,
                        **{f"BC/{key}": value for key, value in metrics.items()},
                    }
                )

        try:
            fit_actor(
                policy,
                bc,
                gradient_steps=args.gradient_steps,
                learning_rate=args.learning_rate,
                final_learning_rate=args.final_learning_rate,
                l2_weight=l2,
                log_interval=args.log_interval,
                on_metrics=record,
            )
            facts.update(actor_fit_metrics(policy, bc, l2_weight=l2))
            facts["fit_trace"] = trace
            if logger is not None:
                facts["wandb_url"] = logger.url
        finally:
            if logger is not None:
                logger.finish()
        save_actor(policy, str(directory / f"{label}.pt"), facts=facts)
        (directory / f"{label}.json").write_text(json.dumps(facts, indent=2) + "\n")
        return policy, facts

    for l2 in candidates:
        for fold, (training, validation) in enumerate(folds):
            _, facts = fit(f"lambda_{l2:g}_fold_{fold}", training, validation, l2)
            records.append({"fold": fold, **facts})
    scores = {
        l2: sum(
            row[f"validation_{args.loss_type}"]
            for row in records
            if row["l2_weight"] == l2
        )
        / len(folds)
        for l2 in candidates
    }
    selected_l2 = min(scores, key=scores.get)
    summary = {
        "selection": selection,
        "fold_scores": records,
        f"mean_validation_{args.loss_type}": scores,
        "selected_l2": selected_l2,
        "output": str(output),
    }
    summary_path = directory / "selection.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[INFO] BC validation: {scores}; selected lambda={selected_l2}", flush=True)
    policy, facts = fit(
        "all_demos_selected", list(range(len(episodes))), [], selected_l2
    )
    facts["selection_file"] = str(summary_path)
    save_actor(policy, str(output), facts=facts)
    # A fully fitted unregularized control distinguishes schedule and L2 effects.
    if selected_l2 != 0.0:
        fit("all_demos_unregularized", list(range(len(episodes))), [], 0.0)
    return summary


def main(argv: list[str] | None = None) -> None:
    run(build_bc_regularization_parser().parse_args(argv))
