"""Fit only the spread of a fully trained MSE actor to interior expert actions."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch

from algorithms.off_policy.actor import (
    StateDependentTanhPolicy,
    policy_observation_tensor,
)
from config import load_experiment_cfg
from config.cli import build_spread_calibration_parser
from data.expert_obs_actions import load_obs_action_episodes
from storage.expert_dataset import ExpertObsActionDataset
from storage.observations import Observation, map_observation
from training.bc_regularization import episode_folds
from training.behavior_cloning import build_actor, dataset_action_mse, save_actor
from training.pipeline import REPO_ROOT


def interior_actions(actions: torch.Tensor, *, bound: float) -> torch.Tensor:
    """Valid tanh-density rows [B]; hard-clipped boundary actions are censored."""
    if actions.ndim != 2 or not torch.isfinite(actions).all():
        raise ValueError(f"Expected finite actions [B,A], got {tuple(actions.shape)}.")
    if bool((actions.abs() > bound * (1 + 1e-6)).any()):
        raise ValueError("Expert action exceeds the policy action bound.")
    return (actions.abs() < bound * (1 - 1e-6)).all(dim=-1)


def load_mse_mean(
    policy: StateDependentTanhPolicy,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> dict:
    """Copy only trunk/mean weights; reject a source that is not a full MSE fit."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    facts = checkpoint["behavior_cloning"]
    if facts.get("loss_type") != "mse" or facts.get("gradient_steps", 0) <= 0:
        raise ValueError("Spread calibration requires a completed MSE BC checkpoint.")
    source = checkpoint["actor_state_dict"]
    for prefix, module in (
        ("trunk", policy.model.trunk),
        ("mean_head", policy.model.mean_head),
    ):
        values = {
            name.removeprefix(prefix + "."): value
            for name, value in source.items()
            if name.startswith(prefix + ".")
        }
        module.load_state_dict(values, strict=True)
        module.requires_grad_(False)
    if not hasattr(policy.model, "std_head"):
        raise ValueError("Spread calibration requires the smooth-bounded std head.")
    return facts


def _interior_pairs(
    observations: Observation, actions: torch.Tensor, *, bound: float
) -> tuple[Observation, torch.Tensor, int]:
    mask = interior_actions(actions, bound=bound)
    if not bool(mask.any()):
        raise ValueError("No interior expert actions remain for spread fitting.")
    return (
        map_observation(observations, lambda leaf: leaf[mask]),
        actions[mask],
        int((~mask).sum()),
    )


@torch.no_grad()
def spread_metrics(
    policy: StateDependentTanhPolicy,
    dataset: ExpertObsActionDataset,
    *,
    device: str | torch.device,
) -> dict[str, float]:
    """Full-dataset interior NLL and deterministic action error; no sampling."""
    observations, actions = dataset.all_pairs(device)
    interior_obs, interior_targets, excluded = _interior_pairs(
        observations, actions, bound=policy.model.action_bound
    )
    _, std = policy.distribution_parameters(interior_obs)
    return {
        "nll_interior": float(
            -policy.log_probability(interior_obs, interior_targets).mean()
        ),
        "action_mse": dataset_action_mse(policy, dataset, device=device),
        "std_mean": float(std.mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "interior_pairs": float(interior_targets.shape[0]),
        "excluded_boundary_pairs": float(excluded),
    }


def fit_spread(
    policy: StateDependentTanhPolicy,
    dataset: ExpertObsActionDataset,
    *,
    gradient_steps: int,
    batch_size: int,
    learning_rate: float,
    final_learning_rate: float,
) -> None:
    """Update only the std head from interior demonstration actions [B,A]."""
    if gradient_steps <= 0 or batch_size <= 0:
        raise ValueError("gradient_steps and batch_size must be positive.")
    if not (0 < final_learning_rate <= learning_rate < math.inf):
        raise ValueError("Require finite 0 < final_learning_rate <= learning_rate.")
    device = next(policy.model.parameters()).device
    all_obs, all_actions = dataset.all_pairs(device)
    _, _, excluded = _interior_pairs(
        all_obs, all_actions, bound=policy.model.action_bound
    )
    if excluded == len(dataset):
        raise ValueError("All expert actions lie on the execution boundary.")
    optimizer = torch.optim.Adam(policy.model.std_head.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=gradient_steps, eta_min=final_learning_rate
    )
    mean_before = {
        name: value.detach().clone()
        for name, value in policy.model.state_dict().items()
        if name.startswith(("trunk.", "mean_head."))
    }
    for _ in range(gradient_steps):
        observations, actions = dataset.sample(batch_size, device)
        interior_obs, interior_targets, _ = _interior_pairs(
            observations, actions, bound=policy.model.action_bound
        )
        loss = -policy.log_probability(interior_obs, interior_targets).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Spread likelihood became nonfinite.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
    for name, value in policy.model.state_dict().items():
        if name in mean_before and not torch.equal(value, mean_before[name]):
            raise RuntimeError(
                f"Mean policy changed during spread calibration: {name}."
            )


def run(args: argparse.Namespace) -> dict:
    """Whole-episode floor selection, then fit each floor on all ten demos."""
    cfg, _ = load_experiment_cfg(args=args, config_root=REPO_ROOT / "configs")
    if cfg.expert_data.num_trajectories is not None:
        raise ValueError("Supply an exact ten-demo file; use all of its episodes.")
    if cfg.init_policy_checkpoint is not None:
        raise ValueError("Give the MSE mean source with --mean_checkpoint.")
    floors = sorted(set(args.std_floors))
    policy_cfg = cfg.algorithm.policy
    if not floors or any(not 0 < floor < policy_cfg.initial_std for floor in floors):
        raise ValueError("std_floors must be in (0, policy.initial_std).")
    if policy_cfg.std_parameterization != "clamped_log":
        raise ValueError("The source experiment must describe the MSE actor.")
    episodes = load_obs_action_episodes(cfg.expert_data.path)
    if len(episodes) != 10:
        raise ValueError(
            f"Expected the study's ten complete demos, got {len(episodes)}."
        )
    folds = episode_folds(len(episodes), 5, seed=cfg.seed)
    expected_training = list(range(len(episodes)))
    full_checkpoint = Path(args.mean_checkpoint).resolve(strict=True)
    source_facts = torch.load(full_checkpoint, map_location="cpu", weights_only=False)[
        "behavior_cloning"
    ]
    if source_facts.get("training_episodes") != expected_training:
        raise ValueError("The MSE source must have been fitted on all supplied demos.")
    if (
        Path(source_facts["expert_data_path"]).resolve()
        != Path(cfg.expert_data.path).resolve()
    ):
        raise ValueError(
            "Mean checkpoint and spread calibration must use one demo file."
        )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    def make_policy(floor: float, source: Path) -> StateDependentTanhPolicy:
        torch.manual_seed(cfg.seed)
        candidate_cfg = replace(
            cfg,
            algorithm=replace(
                cfg.algorithm,
                policy=replace(
                    policy_cfg,
                    std_parameterization="smooth_bounded",
                    minimum_std=floor,
                ),
            ),
        )
        observations, actions = episodes[0]
        actor = build_actor(
            candidate_cfg,
            observation_dim=policy_observation_tensor(observations).shape[-1],
            action_dim=actions.shape[-1],
            device=cfg.device,
        )
        facts = load_mse_mean(actor, source, device=cfg.device)
        if (
            Path(facts["expert_data_path"]).resolve()
            != Path(cfg.expert_data.path).resolve()
        ):
            raise ValueError(
                "Fold mean checkpoint uses a different demonstration file."
            )
        return actor

    records = []
    for floor in floors:
        for fold_index, (training, validation) in enumerate(folds):
            fold_checkpoint = (
                Path(args.fold_checkpoint_dir) / f"lambda_0_fold_{fold_index}.pt"
            )
            policy = make_policy(floor, fold_checkpoint)
            facts = torch.load(fold_checkpoint, map_location="cpu", weights_only=False)[
                "behavior_cloning"
            ]
            if (
                facts.get("l2_weight") != 0
                or facts.get("training_episodes") != training
                or facts.get("validation_episodes") != validation
            ):
                raise ValueError(
                    f"MSE fold {fold_index} does not match the current episode split."
                )
            train_dataset = ExpertObsActionDataset(
                [episodes[index] for index in training]
            )
            validation_dataset = ExpertObsActionDataset(
                [episodes[index] for index in validation]
            )
            fit_spread(
                policy,
                train_dataset,
                gradient_steps=args.gradient_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                final_learning_rate=args.final_learning_rate,
            )
            metrics = spread_metrics(policy, validation_dataset, device=cfg.device)
            record = {
                "floor": floor,
                "fold": fold_index,
                "validation_episodes": validation,
                **metrics,
            }
            records.append(record)
            print(f"[INFO] spread validation: {record}", flush=True)
    scores = {
        floor: sum(row["nll_interior"] for row in records if row["floor"] == floor)
        / len(folds)
        for floor in floors
    }
    selected = min(scores, key=scores.get)
    final = []
    all_demos = ExpertObsActionDataset(episodes)
    for floor in floors:
        policy = make_policy(floor, full_checkpoint)
        fit_spread(
            policy,
            all_demos,
            gradient_steps=args.gradient_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            final_learning_rate=args.final_learning_rate,
        )
        metrics = spread_metrics(policy, all_demos, device=cfg.device)
        path = output / f"floor_{floor:g}.pt"
        facts = {
            "seed": cfg.seed,
            "loss_type": "mse_mean_plus_interior_nll_spread",
            "expert_data_path": str(Path(cfg.expert_data.path).resolve()),
            "mean_checkpoint": str(full_checkpoint),
            "mean_fit_steps": source_facts["gradient_steps"],
            "spread_fit_steps": args.gradient_steps,
            "training_episodes": expected_training,
            "policy": asdict(
                replace(
                    policy_cfg, minimum_std=floor, std_parameterization="smooth_bounded"
                )
            ),
            **metrics,
        }
        save_actor(policy, str(path), facts=facts)
        final.append({"floor": floor, "checkpoint": str(path), **metrics})
        print(f"[INFO] spread final: {final[-1]}", flush=True)
    summary = {
        "selection": "five-fold held-out interior NLL with frozen MSE mean",
        "selected_floor": selected,
        "validation_nll": scores,
        "folds": records,
        "full_fits": final,
        "online_interactions": 0,
    }
    if cfg.logger == "wandb":
        import wandb

        with wandb.init(
            project=cfg.wandb_project,
            name=output.name,
            group="lift_xy06_bc_spread_calibration",
            job_type="offline_bc_spread",
            dir=str(output),
            config={
                "demonstrations": str(Path(cfg.expert_data.path).resolve()),
                "mean_checkpoint": str(full_checkpoint),
                "gradient_steps_per_fit": args.gradient_steps,
                "std_floors": floors,
                "selection": summary["selection"],
            },
        ) as logger:
            logger.define_metric("Calibration/fold", hidden=True)
            logger.define_metric(
                "Calibration/validation_nll", step_metric="Calibration/fold"
            )
            for index, row in enumerate(records):
                logger.log(
                    {
                        "Calibration/fold": index,
                        "Calibration/floor": row["floor"],
                        "Calibration/validation_nll": row["nll_interior"],
                        "Calibration/validation_action_mse": row["action_mse"],
                        "Calibration/validation_std_mean": row["std_mean"],
                        "Calibration/excluded_boundary_pairs": row[
                            "excluded_boundary_pairs"
                        ],
                    }
                )
            for row in final:
                logger.log(
                    {
                        "Calibration/final_floor": row["floor"],
                        "Calibration/final_nll": row["nll_interior"],
                        "Calibration/final_action_mse": row["action_mse"],
                        "Calibration/final_std_mean": row["std_mean"],
                    }
                )
            summary["wandb_url"] = logger.url
    (output / "selection.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(f"[INFO] selected spread floor {selected:g}: {scores}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    run(build_spread_calibration_parser().parse_args(argv))
