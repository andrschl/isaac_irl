"""Fit the online arms' actor to demonstrations, offline and before any run.

Every arm of the study starts from the same behaviour-cloned actor, so cloning
happens once, on its own, and is written to a checkpoint the online runs load.
Nothing here touches a simulator, a critic or a replay buffer: the only thing
that would change across arms is the thing being compared.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from adapters.checkpoints import load_actor_only_checkpoint
from algorithms.bc import BC, build_bc_alg
from algorithms.off_policy.actor import (
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
    policy_observation_tensor,
)
from config import build_behavior_cloning_parser, load_experiment_cfg
from config.schema import BCCfg, PmdCfg, TrainCfg
from interfaces import RuntimeContext
from storage.expert_dataset import ExpertObsActionDataset
from training.pipeline import REPO_ROOT


@dataclass(frozen=True, slots=True)
class BehaviorCloningResult:
    """What one offline fit produced, for the caller to report or assert on."""

    policy: StateDependentTanhPolicy
    gradient_steps: int
    expert_pairs: int
    final_loss: float
    checkpoint_path: str


def build_actor(
    cfg: TrainCfg, *, observation_dim: int, action_dim: int, device: str
) -> StateDependentTanhPolicy:
    """Build the actor the online arms will start from.

    It is the online policy exactly, architecture and bound included, because
    an actor the off-policy learner cannot load is not a starting point.
    """
    if not isinstance(cfg.algorithm, PmdCfg):
        raise ValueError(
            "Offline behavioral cloning fits the off-policy actor; set "
            "algo.type=pmd in the experiment config."
        )
    policy_cfg = cfg.algorithm.policy
    model = StateDependentTanhGaussianActor(
        observation_dim,
        action_dim,
        hidden_dims=policy_cfg.hidden_dims,
        action_bound=policy_cfg.action_bound,
        initial_std=policy_cfg.initial_std,
        minimum_log_std=math.log(policy_cfg.minimum_std),
        maximum_log_std=math.log(policy_cfg.maximum_std),
        std_parameterization=policy_cfg.std_parameterization,
    ).to(device)
    return StateDependentTanhPolicy(model)


def fit_actor(
    policy: StateDependentTanhPolicy,
    behavior_cloning: BC,
    *,
    gradient_steps: int,
    learning_rate: float,
    log_interval: int = 0,
    l2_weight: float = 0.0,
    final_learning_rate: float | None = None,
    on_metrics: Callable[[int, dict[str, float]], None] | None = None,
) -> float:
    """Fit configured BC loss + lambda/2 * sum(W^2); return final minibatch loss.

    W contains the mean-policy weight matrices, not biases or the exploration
    head. Optional cosine LR decay spends the entire specified budget; no
    checkpoint is selected by validation loss or environment success. Metrics
    use all training/held-out pairs, rather than a noisy minibatch.
    """
    if gradient_steps <= 0:
        raise ValueError(f"gradient_steps must be > 0, got {gradient_steps}.")
    if not math.isfinite(l2_weight) or l2_weight < 0.0:
        raise ValueError("l2_weight must be finite and nonnegative.")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive.")
    if final_learning_rate is not None and not (
        0.0 < final_learning_rate <= learning_rate
    ):
        raise ValueError("Require 0 < final_learning_rate <= learning_rate.")
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=float(learning_rate))
    scheduler = (
        None
        if final_learning_rate is None
        else torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=gradient_steps, eta_min=final_learning_rate
        )
    )

    def record(step: int) -> None:
        metrics = actor_fit_metrics(policy, behavior_cloning, l2_weight=l2_weight)
        metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        if on_metrics is not None:
            on_metrics(step, metrics)
        if log_interval > 0:
            print(f"[INFO] bc step {step}/{gradient_steps}: {metrics}", flush=True)

    if log_interval > 0 or on_metrics is not None:
        record(0)
    loss_value = float("nan")
    for step in range(1, gradient_steps + 1):
        loss = behavior_cloning.compute_policy_loss(policy)
        objective = loss + l2_weight * mean_policy_l2(policy) if l2_weight else loss
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_value = float(loss.item())
        if (log_interval > 0 and step % log_interval == 0) or (
            step == gradient_steps and on_metrics is not None
        ):
            record(step)
    return loss_value


def mean_policy_l2(policy: StateDependentTanhPolicy) -> torch.Tensor:
    """Half the squared norm of mean-policy matrices; scalar, with gradients."""
    matrices = [
        parameter
        for module in (policy.model.trunk, policy.model.mean_head)
        for parameter in module.parameters()
        if parameter.ndim > 1
    ]
    return 0.5 * sum(parameter.square().sum() for parameter in matrices)


@torch.no_grad()
def dataset_action_mse(
    policy: StateDependentTanhPolicy,
    dataset: ExpertObsActionDataset,
    *,
    device: str | torch.device,
) -> float:
    """Full-dataset deterministic-action MSE on obs [B,O], actions [B,A]."""
    observations, actions = dataset.all_pairs(device)
    return float(
        torch.nn.functional.mse_loss(policy.deterministic_action(observations), actions)
    )


@torch.no_grad()
def actor_fit_metrics(
    policy: StateDependentTanhPolicy, bc: BC, *, l2_weight: float
) -> dict[str, float]:
    """Full-dataset action error, NLL/spread when fitted, and the actual objective."""
    penalty = float(mean_policy_l2(policy))
    metrics = {
        "weight_half_squared_norm": penalty,
        "regularization": l2_weight * penalty,
    }
    for prefix, dataset in (("training", bc.storage), ("validation", bc.val_storage)):
        if dataset is None:
            continue
        metrics[f"{prefix}_mse"] = dataset_action_mse(policy, dataset, device=bc.device)
        if bc.cfg.loss_type == "nll":
            observations, actions = dataset.all_pairs(bc.device)
            metrics[f"{prefix}_nll"] = float(
                -policy.log_probability(observations, actions).mean()
            )
            _, std = policy.distribution_parameters(observations)
            metrics[f"{prefix}_std_mean"] = float(std.mean())
            metrics[f"{prefix}_std_min"] = float(std.min())
            metrics[f"{prefix}_std_max"] = float(std.max())
    metrics["training_objective"] = (
        metrics[f"training_{bc.cfg.loss_type}"] + l2_weight * penalty
    )
    return metrics


def save_actor(
    policy: StateDependentTanhPolicy, path: str, *, facts: dict[str, Any]
) -> None:
    """Write the actor in the layout ``load_actor_only_checkpoint`` reads.

    ``facts`` travels with it so a run started from this actor can say what it
    started from without reading a log.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {"actor_state_dict": policy.model.state_dict(), "behavior_cloning": facts}, path
    )


def run(args: argparse.Namespace) -> BehaviorCloningResult:
    """Load demonstrations, fit the actor, and write one checkpoint."""
    cfg, _ = load_experiment_cfg(args=args, config_root=REPO_ROOT / "configs")
    torch.manual_seed(int(cfg.seed))
    # Offline cloning is the whole objective here, not a regularizer.
    bc_cfg = BCCfg(
        weight=1.0,
        loss_type=args.loss_type,
        batch_size=int(args.batch_size),
        val_fraction=float(args.validation_fraction),
    )
    behavior_cloning = build_bc_alg(
        cfg=bc_cfg,
        expert_data_path=os.path.abspath(cfg.expert_data.path),
        ctx=RuntimeContext(num_envs=1, feature_dim=1, device=cfg.device),
        device=cfg.device,
        expert_num_trajectories=cfg.expert_data.num_trajectories,
        expert_subset_strategy=cfg.expert_data.subset_strategy,
        seed=cfg.seed,
    )
    if behavior_cloning is None:
        raise RuntimeError("Behavioral cloning was disabled; nothing to fit.")

    observations, actions = behavior_cloning.storage.all_pairs(cfg.device)
    policy = build_actor(
        cfg,
        observation_dim=int(policy_observation_tensor(observations).shape[-1]),
        action_dim=int(actions.shape[-1]),
        device=cfg.device,
    )
    if cfg.init_policy_checkpoint is not None:
        load_actor_only_checkpoint(
            policy.model, cfg.init_policy_checkpoint, device=cfg.device
        )
    trace = []
    final_loss = fit_actor(
        policy,
        behavior_cloning,
        gradient_steps=int(args.gradient_steps),
        learning_rate=float(args.learning_rate),
        log_interval=int(args.log_interval),
        l2_weight=float(args.l2_weight),
        final_learning_rate=args.final_learning_rate,
        on_metrics=lambda step, metrics: trace.append({"step": step, **metrics}),
    )
    facts = {
        "expert_data_path": cfg.expert_data.path,
        "expert_pairs": len(behavior_cloning.storage),
        "gradient_steps": int(args.gradient_steps),
        "learning_rate": float(args.learning_rate),
        "final_learning_rate": args.final_learning_rate,
        "l2_weight": float(args.l2_weight),
        "validation_fraction": float(args.validation_fraction),
        "validation_pairs": 0
        if behavior_cloning.val_storage is None
        else len(behavior_cloning.val_storage),
        "fit_trace": trace,
        **actor_fit_metrics(policy, behavior_cloning, l2_weight=float(args.l2_weight)),
        "batch_size": int(args.batch_size),
        "loss_type": bc_cfg.loss_type,
        "policy": asdict(cfg.algorithm.policy),
        "seed": int(cfg.seed),
        "final_loss": final_loss,
        "init_policy_checkpoint": cfg.init_policy_checkpoint,
    }
    save_actor(policy, args.output, facts=facts)
    print(
        f"[INFO] behavioral cloning finished: {args.gradient_steps} steps over "
        f"{len(behavior_cloning.storage)} expert pairs, final loss "
        f"{final_loss:.6f}, actor written to {args.output}.",
        flush=True,
    )
    return BehaviorCloningResult(
        policy=policy,
        gradient_steps=int(args.gradient_steps),
        expert_pairs=len(behavior_cloning.storage),
        final_loss=final_loss,
        checkpoint_path=str(args.output),
    )


def main(argv: list[str] | None = None) -> None:
    run(build_behavior_cloning_parser().parse_args(argv))
