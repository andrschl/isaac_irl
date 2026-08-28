"""Shared utilities for the SAC and exponential-PMD learners."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Protocol

import torch
import torch.nn as nn
from einops import rearrange

from algorithms.state_dependent_tanh_actor import TanhPolicyProtocol
from algorithms.successor_ensemble import RunningStandardizer
from storage.observations import actor_observation
from storage.transition_replay import (
    Observation,
    TransitionReplayBuffer,
    TransitionRollout,
)


class OffPolicyLearner(Protocol):
    """Algorithm contract driven by the collection adapters.

    Implementations own the actor, the critics and their optimizers, and never
    touch the environment: the adapter hands them replay and asks for updates.
    """

    policy: TanhPolicyProtocol
    policy_version: int

    def observe_rollout(self, rollout: TransitionRollout) -> None: ...

    def critic_update(self, replay: TransitionReplayBuffer) -> dict[str, float]: ...

    def actor_update(self, replay: TransitionReplayBuffer) -> dict[str, float]: ...

    def train(self) -> None: ...

    def eval(self) -> None: ...

    def save_state(self) -> dict[str, Any]: ...

    def load_state(
        self, state: dict[str, Any], *, load_optimizer: bool = True
    ) -> None: ...


class BehaviorCloningLoss(Protocol):
    """Optional expert loss mixed into an actor update."""

    @property
    def weight(self) -> float: ...

    def compute_policy_loss(self, policy: TanhPolicyProtocol) -> torch.Tensor: ...


def update_observation_normalizers(
    rollout: TransitionRollout,
    *,
    actor: RunningStandardizer,
    critic: RunningStandardizer,
) -> None:
    """Update moments from the raw visited observations [N,T,O] of one rollout."""
    actor_values = actor_observation(rollout.policy_observations)
    actor.update(rearrange(actor_values, "n t o -> (n t) o").to(actor.mean.device))
    critic.update(
        rearrange(rollout.critic_observations, "n t o -> (n t) o").to(
            critic.mean.device
        )
    )


def repeat_observations(observations: Observation, repeats: int) -> Observation:
    """Repeat each observation consecutively from [B,...] to [B*K,...]."""
    if repeats <= 0:
        raise ValueError("repeats must be > 0.")
    if isinstance(observations, torch.Tensor):
        return observations.repeat_interleave(repeats, dim=0)
    return {
        key: repeat_observations(value, repeats) for key, value in observations.items()
    }


def mix_actor_loss(
    rl_loss: torch.Tensor,
    *,
    policy: TanhPolicyProtocol,
    bc: BehaviorCloningLoss | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (1-bc_weight)*RL + bc_weight*BC and detached BC loss."""
    if bc is None:
        return rl_loss, rl_loss.new_zeros(())
    weight = float(bc.weight)
    if not 0.0 < weight <= 1.0:
        raise ValueError(f"Active BC weight must be in (0,1], got {weight}.")
    bc_loss = bc.compute_policy_loss(policy)
    if bc_loss.ndim != 0 or not torch.isfinite(bc_loss):
        raise FloatingPointError("BC loss must be a finite scalar.")
    return (1.0 - weight) * rl_loss + weight * bc_loss, bc_loss.detach()


@contextmanager
def frozen_modules(modules: tuple[nn.Module, ...]) -> Iterator[None]:
    """Freeze value parameters while retaining gradients through their inputs."""
    parameters = [parameter for module in modules for parameter in module.parameters()]
    previous = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in zip(parameters, previous, strict=True):
            parameter.requires_grad_(requires_grad)


def latent_policy_kl(
    reference: TanhPolicyProtocol,
    current: TanhPolicyProtocol,
    observations: Observation,
) -> torch.Tensor:
    """Exact KL(reference || current); tanh preserves latent Gaussian KL."""
    reference_mean, reference_std = reference.distribution_parameters(observations)
    current_mean, current_std = current.distribution_parameters(observations)
    reference_dist = torch.distributions.Normal(reference_mean, reference_std)
    current_dist = torch.distributions.Normal(current_mean, current_std)
    return (
        torch.distributions.kl_divergence(reference_dist, current_dist)
        .sum(dim=-1)
        .mean()
    )


def decomposed_policy_kl(
    reference: TanhPolicyProtocol,
    current: TanhPolicyProtocol,
    observations: Observation,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split KL(reference || current) into its mean and spread parts.

    One bound on the whole Gaussian cannot tell "moved the actions" from "made
    the actions noisier": a policy can spend its entire trust region widening
    its standard deviation, a step at a time, and never commit to an action.
    Holding one factor fixed at the reference isolates the other:

        mean part:   sum_i (mu_ref - mu_cur)^2 / (2 sigma_cur^2)
        spread part: sum_i [log(sigma_cur/sigma_ref)
                            + sigma_ref^2 / (2 sigma_cur^2) - 1/2]

    Both are averaged over states and sum to the full KL.
    """
    reference_mean, reference_std = reference.distribution_parameters(observations)
    current_mean, current_std = current.distribution_parameters(observations)
    mean_kl = (
        (reference_mean - current_mean).square() / (2.0 * current_std.square())
    ).sum(dim=-1)
    spread_kl = (
        (current_std / reference_std).log()
        + reference_std.square() / (2.0 * current_std.square())
        - 0.5
    ).sum(dim=-1)
    return mean_kl.mean(), spread_kl.mean()


@torch.no_grad()
def interpolate_parameters(
    module: nn.Module,
    start: dict[str, torch.Tensor],
    proposed: dict[str, torch.Tensor],
    factor: float,
) -> None:
    """Set module parameters to ``start + factor * (proposed - start)``.

    Used to shrink an optimizer displacement that left a trust region, without
    having to re-run or roll back the optimizer state.
    """
    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"factor must be in [0, 1], got {factor}.")
    for name, parameter in module.named_parameters():
        parameter.copy_(start[name] + factor * (proposed[name] - start[name]))


def copy_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def parameter_delta(before: dict[str, torch.Tensor], module: nn.Module) -> float:
    differences = [
        (parameter.detach() - before[name]).flatten()
        for name, parameter in module.named_parameters()
    ]
    return float(torch.cat(differences).norm().item()) if differences else 0.0


def gradient_norm(parameters) -> float:
    gradients = [
        parameter.grad.detach().flatten()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.cat(gradients).norm().item()) if gradients else 0.0
