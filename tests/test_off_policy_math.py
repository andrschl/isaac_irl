from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from algorithms.bc import BCCfg, build_bc_alg
from algorithms.exponential_policy import (
    adaptive_exponential_target,
    exponential_log_weights,
    exponential_target,
)
from algorithms.sac import Sac
from algorithms.soft_value import entropy_return_td_target, soft_task_q_td_target
from algorithms.successor_ensemble import (
    SuccessorFeatureEnsemble,
    estimate_successor_return_scale,
    successor_ensemble_td_target,
    successor_return_scale_from_second_moment,
)
from config.off_policy import OffPolicyTrainCfg, SacCfg
from reward_model.fixed import FixedLinearReward
from utils.runtime_context import RuntimeContext


def test_exponential_target_uses_proposal_correction() -> None:
    q_values = torch.tensor([[0.0, 0.5, 1.0]])
    log_pi_old = torch.tensor([[-0.2, -1.1, -2.0]])

    log_weights = exponential_log_weights(
        q_values,
        log_pi_old,
        beta=0.25,
        q_coefficient=2.0,
    )
    expected = -0.75 * log_pi_old + 2.0 * q_values
    expected = expected - expected.max(dim=1, keepdim=True).values

    torch.testing.assert_close(log_weights, expected)


def test_beta_endpoints_match_boltzmann_and_mirror_targets() -> None:
    q_values = torch.tensor([[0.1, 0.4, -0.2]])
    log_pi_old = torch.tensor([[-0.3, -0.8, -1.7]])

    sac_family = exponential_target(q_values, log_pi_old, beta=0.0, q_coefficient=3.0)
    mirror = exponential_target(q_values, log_pi_old, beta=1.0, q_coefficient=3.0)

    torch.testing.assert_close(
        sac_family.probabilities,
        torch.softmax(-log_pi_old + 3.0 * q_values, dim=1),
    )
    torch.testing.assert_close(
        mirror.probabilities,
        torch.softmax(3.0 * q_values, dim=1),
    )


def test_adaptive_coefficient_respects_candidate_kl() -> None:
    q_values = torch.tensor([[-1.0, -0.2, 0.4, 1.1], [0.3, -0.7, 0.8, -0.1]])
    log_pi_old = torch.zeros_like(q_values)

    target = adaptive_exponential_target(
        q_values,
        log_pi_old,
        beta=1.0,
        target_kl=0.05,
        coefficient_min=1.0e-6,
        coefficient_max=1.0e3,
        bisection_steps=48,
    )

    assert 0.0499 <= target.candidate_kl <= 0.05
    torch.testing.assert_close(target.weights.mean(dim=1), torch.ones(2))
    assert math.isfinite(target.q_coefficient)


def test_reward_scale_is_not_normalized_away() -> None:
    q_values = torch.tensor([[0.0, 0.2, 0.7]])
    log_pi_old = torch.zeros_like(q_values)
    small = exponential_target(q_values, log_pi_old, beta=1.0, q_coefficient=1.0)
    large = exponential_target(10.0 * q_values, log_pi_old, beta=1.0, q_coefficient=1.0)

    assert large.probabilities.max() > small.probabilities.max()


def test_successor_target_is_reward_and_entropy_independent() -> None:
    features = torch.tensor([[1.0, -0.5], [0.2, 0.8]])
    next_successors = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
    bootstrap = torch.tensor([True, False])

    target = successor_ensemble_td_target(
        features, next_successors, bootstrap, gamma=0.9
    )

    torch.testing.assert_close(target, torch.tensor([[2.8, 2.2], [0.2, 0.8]]))


def test_entropy_and_direct_soft_targets_share_sac_semantics() -> None:
    next_log_pi = torch.tensor([-2.0, 0.5])
    next_entropy_q = torch.tensor([3.0, 4.0])
    bootstrap = torch.tensor([True, False])
    entropy_target = entropy_return_td_target(
        next_log_pi, next_entropy_q, bootstrap, gamma=0.9
    )
    direct_target = soft_task_q_td_target(
        torch.tensor([1.0, 2.0]),
        next_log_pi,
        torch.tensor([3.0, 4.0]),
        bootstrap,
        gamma=0.9,
        entropy_coefficient=0.2,
    )

    torch.testing.assert_close(entropy_target, torch.tensor([4.5, 0.0]))
    torch.testing.assert_close(direct_target, torch.tensor([4.06, 2.0]))


def test_successor_ensemble_has_explicit_head_batch_feature_shape() -> None:
    ensemble = SuccessorFeatureEnsemble(
        3,
        2,
        4,
        num_heads=5,
        width=8,
        residual_blocks=1,
        latent_dim=6,
        head_hidden_dim=7,
    )
    observations = torch.randn(11, 3)
    actions = torch.randn(11, 2)

    assert ensemble(observations, actions).shape == (5, 11, 4)
    assert ensemble.head(2, observations, actions).shape == (11, 4)
    assert ensemble.mean(observations, actions).shape == (11, 4)
    assert ensemble.heads[0] is not ensemble.heads[1]


def test_successor_return_scale_is_fixed_per_feature() -> None:
    features = torch.tensor([[1.0, 0.0], [-1.0, 2.0]])
    scale = estimate_successor_return_scale(features, gamma=0.5, minimum_scale=1.0e-3)

    expected_rms = features.square().mean(dim=0).sqrt()
    torch.testing.assert_close(scale, expected_rms / 0.5)


def test_fixed_linear_reward_has_no_trainable_parameters() -> None:
    reward = FixedLinearReward(
        ("alive", "upright"), torch.tensor([1.0, -0.5]), step_dt=0.02
    )
    features = torch.tensor([[2.0, 4.0]])

    assert list(reward.parameters()) == []
    torch.testing.assert_close(reward.score(features), torch.tensor([0.0]))


def test_update_to_data_ratio_is_measured_in_transitions() -> None:
    cfg = OffPolicyTrainCfg(updates_per_transition=0.25)

    assert cfg.critic_updates_for(2_048) == 512
    with pytest.raises(ValueError, match="transition_count"):
        cfg.critic_updates_for(0)


def test_zero_bc_weight_constructs_no_loader_or_loss() -> None:
    bc = build_bc_alg(
        cfg=BCCfg(weight=0.0),
        expert_data_path="",
        ctx=RuntimeContext(num_envs=2, feature_dim=3, device="cpu"),
        device="cpu",
    )

    assert bc is None


class _MinimalPolicy:
    def __init__(self) -> None:
        self.model = nn.Linear(2, 1)

    def parameters(self):
        return self.model.parameters()


def test_sac_constructs_no_previous_policy_snapshot() -> None:
    policy = _MinimalPolicy()
    cfg = SacCfg(
        initial_entropy_temperature=0.0,
        automatic_entropy_tuning=False,
    )
    learner = Sac(
        policy,
        object(),
        cfg=cfg,
        action_dim=1,
        device=torch.device("cpu"),
    )

    assert not hasattr(learner, "previous_policy")


def test_output_scale_refit_preserves_successor_predictions() -> None:
    ensemble = SuccessorFeatureEnsemble(
        3,
        2,
        4,
        num_heads=2,
        width=8,
        residual_blocks=1,
        latent_dim=6,
        head_hidden_dim=5,
    )
    for head in ensemble.heads:
        torch.nn.init.normal_(head[-1].weight, std=0.3)
        torch.nn.init.normal_(head[-1].bias, std=0.3)
    observations = torch.randn(7, 3)
    actions = torch.randn(7, 2)
    before = ensemble(observations, actions).clone()

    ensemble.set_output_scale(
        torch.tensor([5.0, 0.5, 20.0, 1.0]), preserve_predictions=True
    )

    torch.testing.assert_close(
        ensemble(observations, actions), before, atol=1.0e-5, rtol=1.0e-5
    )
    torch.testing.assert_close(
        ensemble.output_scale, torch.tensor([5.0, 0.5, 20.0, 1.0])
    )


def test_output_scale_without_preservation_rescales_predictions() -> None:
    ensemble = SuccessorFeatureEnsemble(
        3,
        2,
        2,
        num_heads=1,
        width=8,
        residual_blocks=1,
        latent_dim=6,
        head_hidden_dim=5,
    )
    torch.nn.init.constant_(ensemble.heads[0][-1].bias, 1.0)
    observations = torch.randn(4, 3)
    actions = torch.randn(4, 2)
    before = ensemble(observations, actions).clone()

    ensemble.set_output_scale(torch.tensor([3.0, 3.0]))

    torch.testing.assert_close(ensemble(observations, actions), 3.0 * before)


def test_return_scale_tracks_features_that_only_appear_later() -> None:
    early = torch.zeros(64, 2)
    early[:, 0] = 1.0
    late = torch.ones(64, 2)

    early_scale = estimate_successor_return_scale(early, gamma=0.99, minimum_scale=1.0)
    combined_second_moment = torch.cat((early, late)).square().mean(dim=0)
    later_scale = successor_return_scale_from_second_moment(
        combined_second_moment, gamma=0.99, minimum_scale=1.0
    )

    # The absent feature sits at the floor until it is observed at all.
    assert float(early_scale[1].item()) == pytest.approx(1.0)
    assert float(later_scale[1].item()) > 50.0


def test_apply_head_matches_recomputing_the_trunk() -> None:
    ensemble = SuccessorFeatureEnsemble(
        3,
        2,
        4,
        num_heads=3,
        width=8,
        residual_blocks=1,
        latent_dim=6,
        head_hidden_dim=5,
    )
    observations = torch.randn(9, 3)
    actions = torch.randn(9, 2)

    latent = ensemble.latent(observations, actions)

    for head_index in range(3):
        torch.testing.assert_close(
            ensemble.apply_head(head_index, latent),
            ensemble.head(head_index, observations, actions),
        )


def test_control_reduction_selects_mean_or_pessimistic_minimum() -> None:
    from algorithms.value_backends import _reduce_heads

    per_head = torch.tensor([[1.0, 4.0], [3.0, 2.0], [8.0, 0.0]])

    torch.testing.assert_close(
        _reduce_heads(per_head, "mean"), torch.tensor([4.0, 2.0])
    )
    torch.testing.assert_close(_reduce_heads(per_head, "min"), torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match="Unknown control reduction"):
        _reduce_heads(per_head, "median")


def test_control_reduction_is_configured_not_hardcoded() -> None:
    from config.off_policy import DirectQCfg, SuccessorQCfg

    assert DirectQCfg().control_reduction == "mean"
    assert SuccessorQCfg(control_reduction="min").control_reduction == "min"
    with pytest.raises(ValueError, match="control_reduction"):
        DirectQCfg(control_reduction="min-of-two")


def test_bro_critic_is_layer_normalized_and_residual() -> None:
    from algorithms.bro_network import BroNet
    from algorithms.scalar_action_value import ScalarActionValue

    critic = ScalarActionValue(3, 2, width=16, residual_blocks=2)
    network = critic.network

    assert isinstance(network, BroNet)
    assert len(network.blocks) == 2
    layer_norms = [
        m for m in network.modules() if isinstance(torch.nn.LayerNorm(1), type(m))
    ]
    # stem + two per residual block
    assert len(layer_norms) == 5
    assert critic(torch.randn(7, 3), torch.randn(7, 2)).shape == (7,)


def test_a_bro_critic_does_not_explode_far_off_distribution() -> None:
    """LayerNorm is here to bound extrapolation; check it actually does."""
    from algorithms.scalar_action_value import ScalarActionValue

    torch.manual_seed(0)
    critic = ScalarActionValue(3, 2, width=32, residual_blocks=2)
    observations = torch.randn(64, 3)
    actions = torch.randn(64, 2)

    on_distribution = critic(observations, actions).abs().max()
    far_away = critic(1_000.0 * observations, 1_000.0 * actions).abs().max()

    assert float(far_away) < 20.0 * float(on_distribution) + 10.0


def test_reset_reinitializes_every_head() -> None:
    from algorithms.scalar_action_value import ScalarActionValueEnsemble

    torch.manual_seed(0)
    ensemble = ScalarActionValueEnsemble(3, 2, num_heads=3, width=16, residual_blocks=1)
    observations = torch.randn(5, 3)
    actions = torch.randn(5, 2)
    before = ensemble(observations, actions).clone()

    ensemble.reset_parameters()

    assert not torch.allclose(ensemble(observations, actions), before)


def test_successor_reset_keeps_the_output_scale_and_zero_heads() -> None:
    ensemble = SuccessorFeatureEnsemble(
        3,
        2,
        4,
        num_heads=2,
        width=16,
        residual_blocks=1,
        latent_dim=6,
        head_hidden_dim=5,
    )
    ensemble.set_output_scale(torch.tensor([2.0, 3.0, 4.0, 5.0]))
    for head in ensemble.heads:
        torch.nn.init.normal_(head[-1].weight, std=0.5)

    ensemble.reset_parameters()

    torch.testing.assert_close(
        ensemble.output_scale, torch.tensor([2.0, 3.0, 4.0, 5.0])
    )
    # A freshly reset head predicts no successor return, as at construction.
    predictions = ensemble(torch.randn(4, 3), torch.randn(4, 2))
    torch.testing.assert_close(predictions, torch.zeros_like(predictions))
