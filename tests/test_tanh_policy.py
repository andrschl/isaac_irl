from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from algorithms.off_policy.actor import (  # noqa: E402
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
)
from algorithms.off_policy.common import decomposed_policy_kl  # noqa: E402


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


def _smooth_actor(minimum_std=0.005):
    return StateDependentTanhGaussianActor(
        3,
        2,
        hidden_dims=(8,),
        initial_std=0.1,
        minimum_log_std=math.log(minimum_std),
        maximum_log_std=math.log(2.0),
        std_parameterization="smooth_bounded",
    )


def test_smooth_std_initialization_and_gradients_near_floor():
    actor = _smooth_actor()
    obs = torch.zeros(4, 3)
    _, std = actor.distribution_parameters(obs)
    torch.testing.assert_close(std, torch.full((4, 2), 0.1))
    with torch.no_grad():
        actor.std_head.bias.fill_(-8)
    _, std = actor.distribution_parameters(obs)
    assert bool((std > 0.005).all()) and bool((std < 2).all())
    std.sum().backward()
    assert bool((actor.std_head.bias.grad > 0).all())


def test_smooth_std_checkpoint_roundtrip_and_semantics_guard():
    actor = _smooth_actor()
    clone = _smooth_actor()
    clone.load_state_dict(actor.state_dict())
    obs = torch.randn(4, 3)
    for expected, actual in zip(
        actor.distribution_parameters(obs), clone.distribution_parameters(obs)
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    with pytest.raises(ValueError, match="std bounds differ"):
        _smooth_actor(0.01).load_state_dict(actor.state_dict())
    legacy = _policy().model
    with pytest.raises(RuntimeError):
        actor.load_state_dict(legacy.state_dict())
    with pytest.raises(RuntimeError):
        legacy.load_state_dict(actor.state_dict())


def test_nll_backpropagates_to_mean_and_smooth_spread():
    policy = StateDependentTanhPolicy(_smooth_actor())
    loss = -policy.log_probability(torch.randn(6, 3), torch.zeros(6, 2)).mean()
    loss.backward()
    assert policy.model.mean_head.weight.grad.norm() > 0
    assert policy.model.std_head.weight.grad.norm() > 0


@pytest.mark.parametrize("action_bound", [1.0, 2.0, 4.0])
def test_log_probability_matches_the_change_of_variables_density(
    action_bound: float,
) -> None:
    policy = _policy(action_bound=action_bound)
    observations = torch.randn(5, 3)

    actions, latent_actions, log_probability = policy.sample_with_latent(observations)
    mean, std = policy.distribution_parameters(observations)
    gaussian = torch.distributions.Normal(mean, std)
    # log p(a) = log p(u) - sum_i [log bound + log(1 - tanh(u_i)^2)]
    expected = (
        gaussian.log_prob(latent_actions)
        - math.log(action_bound)
        - torch.log(1.0 - torch.tanh(latent_actions).square())
    ).sum(dim=-1)

    torch.testing.assert_close(log_probability, expected, atol=1.0e-4, rtol=1.0e-4)
    torch.testing.assert_close(actions, action_bound * torch.tanh(latent_actions))
    assert torch.all(actions.abs() <= action_bound)


@pytest.mark.parametrize("action_bound", [1.0, 4.0])
def test_latent_log_probability_matches_the_inverted_one_inside_the_bound(
    action_bound: float,
) -> None:
    policy = _policy(action_bound=action_bound)
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
    assert (
        float(sum(decomposed_policy_kl(snapshot, policy, observations)).item()) == 0.0
    )

    with torch.no_grad():
        policy.model.mean_head.bias.add_(0.5)

    assert float(sum(decomposed_policy_kl(snapshot, policy, observations)).item()) > 0.0
    assert all(not parameter.requires_grad for parameter in snapshot.parameters())


@pytest.mark.parametrize("action_bound", [0.5, 4.0])
def test_sampled_actions_stay_inside_the_bound(action_bound: float) -> None:
    policy = _policy(action_bound=action_bound)
    observations = torch.randn(128, 3)

    actions, _ = policy.sample_with_log_probability(observations)

    assert torch.all(actions.abs() <= action_bound)
