from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from algorithms.off_policy_common import latent_policy_kl  # noqa: E402
from algorithms.state_dependent_tanh_actor import (  # noqa: E402
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
)


def _policy(*, action_bound: float = 1.0) -> StateDependentTanhPolicy:
    torch.manual_seed(0)
    model = StateDependentTanhGaussianActor(
        3,
        2,
        hidden_dims=(8,),
        action_bound=action_bound,
        initial_std=0.5,
    )
    return StateDependentTanhPolicy(model)


def test_log_probability_matches_the_change_of_variables_density() -> None:
    policy = _policy(action_bound=2.0)
    observations = torch.randn(5, 3)

    actions, latent_actions, log_probability = policy.sample_with_latent(observations)
    mean, std = policy.distribution_parameters(observations)
    gaussian = torch.distributions.Normal(mean, std)
    # log p(a) = log p(u) - sum_i [log bound + log(1 - tanh(u_i)^2)]
    expected = (
        gaussian.log_prob(latent_actions)
        - math.log(2.0)
        - torch.log(1.0 - torch.tanh(latent_actions).square())
    ).sum(dim=-1)

    torch.testing.assert_close(log_probability, expected, atol=1.0e-4, rtol=1.0e-4)
    assert torch.all(actions.abs() <= 2.0)


def test_latent_log_probability_matches_the_inverted_one_inside_the_bound() -> None:
    policy = _policy()
    observations = torch.randn(4, 3)
    _, latent_actions, _ = policy.sample_with_latent(observations)
    interior_actions = policy.model.squash(latent_actions.clamp(-1.0, 1.0))
    interior_latents = torch.atanh(interior_actions / policy.model.action_bound)

    from_latent = policy.log_probability_from_latent(observations, interior_latents)
    from_actions = policy.log_probability(observations, interior_actions)

    torch.testing.assert_close(from_latent, from_actions, atol=1.0e-4, rtol=1.0e-4)


def test_snapshot_freezes_the_weights_it_was_taken_from() -> None:
    policy = _policy()
    snapshot = policy.snapshot()
    observations = torch.randn(6, 3)

    torch.testing.assert_close(
        snapshot.deterministic_action(observations),
        policy.deterministic_action(observations),
    )
    assert float(latent_policy_kl(snapshot, policy, observations).item()) == 0.0

    with torch.no_grad():
        policy.model.mean_head.bias.add_(0.5)

    assert float(latent_policy_kl(snapshot, policy, observations).item()) > 0.0
    assert all(not parameter.requires_grad for parameter in snapshot.parameters())


def test_sampled_actions_stay_inside_the_bound() -> None:
    policy = _policy(action_bound=0.5)
    observations = torch.randn(128, 3)

    actions, _ = policy.sample_with_log_probability(observations)

    assert torch.all(actions.abs() <= 0.5)
