"""Exponential policy targets used by policy mirror descent.

For a frozen reference policy pi_old, PMD defines

    pi_target(a|s) proportional to pi_old(a|s)^beta * exp(c * Q(s,a)).

Candidate actions are sampled from pi_old. Therefore the density ratio used
for candidate-weighted maximum likelihood is

    log(pi_target / pi_old) = (beta - 1) * log_pi_old + c * Q + constant.

The implementation keeps this proposal correction explicit; omitting it is
wrong when beta is not one. beta=0 produces the same Boltzmann target that
underlies SAC, while beta=1 is the usual mirror-descent prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_PROBABILITY_FLOOR = 1.0e-12
"""Keeps log(0) out of the entropy of a fully collapsed target."""


@dataclass(frozen=True, slots=True)
class ExponentialTarget:
    """One nonparametric target over K candidates per state.

    Shapes:
        weights: [B,K] normalized to mean one, so a weighted NLL keeps scale
        probabilities: [B,K] the target distribution over the candidates

    The scalars describe how far that target sits from the policy that
    proposed the candidates: ``candidate_kl`` is the realized KL,
    ``q_coefficient`` the Q scaling a bisection chose to reach it, and
    ``effective_sample_size`` how many candidates actually carry weight.
    """

    weights: torch.Tensor
    probabilities: torch.Tensor
    q_coefficient: float
    candidate_kl: float
    effective_sample_size: float

    @property
    def entropy(self) -> float:
        """Entropy of the target over candidates, in nats.

        A collapsed target - all mass on one candidate - is what a Q
        coefficient that is too large looks like from the outside.
        """
        probabilities = self.probabilities
        return float(
            -(probabilities * probabilities.clamp_min(_PROBABILITY_FLOOR).log())
            .sum(dim=1)
            .mean()
            .item()
        )


def exponential_log_weights(
    scalar_q: torch.Tensor,
    reference_log_probability: torch.Tensor,
    *,
    beta: float,
    q_coefficient: float,
) -> torch.Tensor:
    """Return centered log density ratios for candidates [B,K]."""
    _validate_inputs(scalar_q, reference_log_probability)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    if q_coefficient < 0.0:
        raise ValueError("q_coefficient must be >= 0.")
    log_weights = (beta - 1.0) * reference_log_probability + q_coefficient * scalar_q
    return log_weights - log_weights.max(dim=1, keepdim=True).values


def exponential_target(
    scalar_q: torch.Tensor,
    reference_log_probability: torch.Tensor,
    *,
    beta: float,
    q_coefficient: float,
) -> ExponentialTarget:
    """Build normalized target weights and diagnostics for candidates [B,K]."""
    log_weights = exponential_log_weights(
        scalar_q,
        reference_log_probability,
        beta=beta,
        q_coefficient=q_coefficient,
    )
    log_probabilities = torch.log_softmax(log_weights, dim=1)
    probabilities = log_probabilities.exp()
    candidate_count = scalar_q.shape[1]
    log_candidate_count = scalar_q.new_tensor(float(candidate_count)).log()
    candidate_kl = (
        (probabilities * (log_probabilities + log_candidate_count)).sum(dim=1).mean()
    )
    effective_sample_size = probabilities.square().sum(dim=1).reciprocal().mean()
    return ExponentialTarget(
        weights=probabilities * candidate_count,
        probabilities=probabilities,
        q_coefficient=float(q_coefficient),
        candidate_kl=float(candidate_kl.item()),
        effective_sample_size=float(effective_sample_size.item()),
    )


def adaptive_exponential_target(
    scalar_q: torch.Tensor,
    reference_log_probability: torch.Tensor,
    *,
    beta: float,
    target_kl: float,
    coefficient_min: float,
    coefficient_max: float,
    bisection_steps: int,
) -> ExponentialTarget:
    """Solve the Q coefficient so candidate KL is at most target_kl.

    The KL is measured from the categorical target over sampled candidates to
    their uniform proposal indices. This is the nonparametric target KL; the
    separately fitted neural actor has its own projection-KL constraint.
    """
    if target_kl <= 0.0:
        raise ValueError("target_kl must be > 0.")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1].")
    if not 0.0 < coefficient_min < coefficient_max:
        raise ValueError("Expected 0 < coefficient_min < coefficient_max.")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be > 0.")

    lower = exponential_target(
        scalar_q,
        reference_log_probability,
        beta=beta,
        q_coefficient=coefficient_min,
    )
    if lower.candidate_kl >= target_kl:
        return lower
    upper = exponential_target(
        scalar_q,
        reference_log_probability,
        beta=beta,
        q_coefficient=coefficient_max,
    )
    if upper.candidate_kl <= target_kl:
        return upper

    lower_coefficient = coefficient_min
    upper_coefficient = coefficient_max
    selected = lower
    for _ in range(bisection_steps):
        midpoint = 0.5 * (lower_coefficient + upper_coefficient)
        candidate = exponential_target(
            scalar_q,
            reference_log_probability,
            beta=beta,
            q_coefficient=midpoint,
        )
        if candidate.candidate_kl > target_kl:
            upper_coefficient = midpoint
        else:
            lower_coefficient = midpoint
            selected = candidate
    return selected


def _validate_inputs(
    scalar_q: torch.Tensor, reference_log_probability: torch.Tensor
) -> None:
    if scalar_q.ndim != 2 or scalar_q.shape[1] < 2:
        raise ValueError(f"Expected scalar_q [B,K] with K>=2, got {scalar_q.shape}.")
    if reference_log_probability.shape != scalar_q.shape:
        raise ValueError(
            "scalar_q and reference_log_probability must have matching [B,K] shapes."
        )
    if not torch.isfinite(scalar_q).all():
        raise FloatingPointError("scalar_q contains non-finite values.")
    if not torch.isfinite(reference_log_probability).all():
        raise FloatingPointError(
            "reference_log_probability contains non-finite values."
        )
