from __future__ import annotations

import math

import pytest
import torch
from conftest import build_rollout

from algorithms.bc import BCCfg, build_bc_alg
from algorithms.off_policy.exponential_target import (
    adaptive_exponential_target,
    exponential_log_weights,
    exponential_target,
)
from algorithms.off_policy.values.networks import (
    SuccessorFeatureEnsemble,
    successor_ensemble_td_target,
    successor_return_scale_from_second_moment,
    successor_return_span,
)
from algorithms.off_policy.values.targets import (
    entropy_return_td_target,
    soft_task_q_td_target,
)
from config.schema import OffPolicyTrainCfg
from interfaces import RuntimeContext
from reward.fixed import FixedLinearReward


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

    boltzmann = exponential_target(q_values, log_pi_old, beta=0.0, q_coefficient=3.0)
    mirror = exponential_target(q_values, log_pi_old, beta=1.0, q_coefficient=3.0)

    torch.testing.assert_close(
        boltzmann.probabilities,
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


def test_entropy_and_direct_soft_targets_share_max_entropy_semantics() -> None:
    next_log_pi = torch.tensor([-2.0, 0.5])
    next_entropy_q = torch.tensor([3.0, 4.0])
    bootstrap = torch.tensor([True, False])
    entropy_target = entropy_return_td_target(
        next_log_pi, next_entropy_q, bootstrap, gamma=0.9, step_dt=0.5
    )
    direct_target = soft_task_q_td_target(
        torch.tensor([1.0, 2.0]),
        next_log_pi,
        torch.tensor([3.0, 4.0]),
        bootstrap,
        gamma=0.9,
        entropy_rate=0.2,
        step_dt=0.5,
    )

    torch.testing.assert_close(entropy_target, torch.tensor([3.6, 0.0]))
    torch.testing.assert_close(direct_target, torch.tensor([3.88, 2.0]))


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

    latent = ensemble.latent(observations, actions)

    assert ensemble(observations, actions).shape == (5, 11, 4)
    assert ensemble.apply_head(2, latent).shape == (11, 4)
    assert ensemble.heads[0] is not ensemble.heads[1]


def test_successor_return_scale_is_fixed_per_feature() -> None:
    features = torch.tensor([[1.0, 0.0], [-1.0, 2.0]])
    scale = successor_return_scale_from_second_moment(
        features.square().mean(dim=0), gamma=0.5, minimum_scale=1.0e-3
    )

    expected_rms = features.square().mean(dim=0).sqrt()
    torch.testing.assert_close(scale, expected_rms / 0.5)


def test_an_undiscounted_successor_sum_is_scaled_by_the_episode() -> None:
    """Nothing decays at gamma=1, so the episode is what bounds the sum.

    A finite-horizon task has a perfectly good scale - H steps of features -
    but it is not 1/(1-gamma), and reading that limit at gamma=1 gives infinity.
    """
    features = torch.tensor([[1.0, 0.0], [-1.0, 2.0]])
    second_moment = features.square().mean(dim=0)

    scale = successor_return_scale_from_second_moment(
        second_moment, gamma=1.0, minimum_scale=1.0e-3, horizon=250
    )

    torch.testing.assert_close(scale, second_moment.sqrt() * 250.0)
    assert successor_return_span(0.98) == pytest.approx(50.0)
    assert successor_return_span(1.0, 250) == 250.0


def test_gamma_one_without_a_horizon_is_refused() -> None:
    with pytest.raises(ValueError, match="positive episode horizon"):
        successor_return_scale_from_second_moment(
            torch.ones(2), gamma=1.0, minimum_scale=1.0
        )


def test_fixed_linear_reward_has_no_trainable_parameters() -> None:
    reward = FixedLinearReward(
        ("alive", "upright"), torch.tensor([1.0, -0.5]), step_dt=0.02
    )
    features = torch.tensor([[2.0, 4.0]])

    assert list(reward.parameters()) == []
    torch.testing.assert_close(reward.score(features), torch.tensor([0.0]))


def test_episode_and_compute_budgets_are_explicit() -> None:
    cfg = OffPolicyTrainCfg(episodes_per_batch=16, critic_updates_per_batch=512)
    assert cfg.episodes_per_batch == 16
    assert cfg.critic_updates_per_batch == 512
    with pytest.raises(ValueError, match="critic_updates_per_batch"):
        OffPolicyTrainCfg(critic_updates_per_batch=0)


def test_zero_bc_weight_constructs_no_loader_or_loss() -> None:
    bc = build_bc_alg(
        cfg=BCCfg(weight=0.0),
        expert_data_path="",
        ctx=RuntimeContext(num_envs=2, feature_dim=3, device="cpu"),
        device="cpu",
    )

    assert bc is None


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

    early_scale = successor_return_scale_from_second_moment(
        early.square().mean(dim=0), gamma=0.99, minimum_scale=1.0
    )
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
    per_head = ensemble(observations, actions)

    for head_index in range(3):
        torch.testing.assert_close(
            ensemble.apply_head(head_index, latent), per_head[head_index]
        )


def test_control_reduction_selects_mean_or_pessimistic_minimum() -> None:
    from algorithms.off_policy.values.backends import _reduce_heads

    per_head = torch.tensor([[1.0, 4.0], [3.0, 2.0], [8.0, 0.0]])

    torch.testing.assert_close(
        _reduce_heads(per_head, "mean"), torch.tensor([4.0, 2.0])
    )
    torch.testing.assert_close(_reduce_heads(per_head, "min"), torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError, match="Unknown control reduction"):
        _reduce_heads(per_head, "median")


def test_direct_control_reduction_is_configured_not_hardcoded() -> None:
    from config.schema import DirectQCfg

    assert DirectQCfg(control_reduction="mean").control_reduction == "mean"
    with pytest.raises(ValueError, match="control_reduction"):
        DirectQCfg(control_reduction="min-of-two")


def test_successor_control_quantile_is_percentage_based() -> None:
    from config.schema import SuccessorQCfg

    cfg = SuccessorQCfg()
    assert cfg.control_quantile == 0.25


def test_ensemble_quantile_selects_the_lower_quartile_per_batch_item() -> None:
    from algorithms.off_policy.values.backends import _ensemble_quantile

    per_head = torch.tensor(
        [[9.0, 0.0], [1.0, 8.0], [7.0, 2.0], [3.0, 6.0], [5.0, 4.0]]
    )
    expected = torch.tensor([3.0, 2.0])

    torch.testing.assert_close(_ensemble_quantile(per_head, 0.25), expected)
    torch.testing.assert_close(
        _ensemble_quantile(per_head[[3, 0, 4, 1, 2]], 0.25), expected
    )


def test_ensemble_quantile_interpolates_and_handles_endpoints() -> None:
    from algorithms.off_policy.values.backends import _ensemble_quantile

    four_heads = torch.tensor([[0.0], [4.0], [8.0], [12.0]])
    torch.testing.assert_close(
        _ensemble_quantile(four_heads, 0.25), torch.tensor([3.0])
    )
    torch.testing.assert_close(_ensemble_quantile(four_heads, 0.0), torch.tensor([0.0]))
    torch.testing.assert_close(_ensemble_quantile(four_heads, 0.5), torch.tensor([6.0]))

    one_head = torch.tensor([[7.0, -2.0]])
    torch.testing.assert_close(_ensemble_quantile(one_head, 0.25), one_head[0])


@pytest.mark.parametrize(
    "quantile, expected",
    [
        (0.25, [2.0, 4.0]),
        (0.5, [4.0, 6.0]),
        (0.75, [8.0, 7.0]),
        (0.9, [10.4, 7.6]),
        (1.0, [12.0, 8.0]),
    ],
)
def test_three_head_optimism_sweep_interpolates_scalar_scores(quantile, expected):
    from algorithms.off_policy.values.backends import _ensemble_quantile

    # Scalarized Q scores [M=3, B=2], deliberately not sorted by head.
    per_head = torch.tensor([[12.0, 2.0], [0.0, 8.0], [4.0, 6.0]])
    torch.testing.assert_close(
        _ensemble_quantile(per_head, quantile), torch.tensor(expected)
    )


@pytest.mark.parametrize("quantile", [-0.1, 1.1])
def test_successor_control_quantile_rejects_values_outside_the_unit_range(
    quantile: float,
) -> None:
    from config.schema import SuccessorQCfg

    with pytest.raises(ValueError, match="control_quantile"):
        SuccessorQCfg(control_quantile=quantile)


@pytest.mark.parametrize(
    "quantile, expected_reward_q", [(0.25, -0.5), (0.9, 1.6), (1.0, 2.0)]
)
def test_successor_quantile_is_applied_after_signed_reward_scalarization(
    quantile: float, expected_reward_q: float
) -> None:
    from algorithms.off_policy.values.backends import SuccessorActionValue

    successors = torch.tensor(
        [
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[2.0, 0.0]],
            [[0.0, 2.0]],
            [[1.0, 1.0]],
        ]
    )
    backend = SuccessorActionValue.__new__(SuccessorActionValue)
    backend.online = lambda observations, actions: successors
    backend.reward = FixedLinearReward(
        ("positive", "negative"), torch.tensor([2.0, -1.0]), step_dt=0.5
    )
    backend.control_quantile = quantile
    backend.entropy = None

    estimate = backend.mean(torch.zeros(1, 3), torch.zeros(1, 2), entropy_rate=0.0)

    torch.testing.assert_close(estimate.reward_q, torch.tensor([expected_reward_q]))
    torch.testing.assert_close(estimate.q, estimate.reward_q)
    torch.testing.assert_close(estimate.entropy_q, torch.zeros(1))


def test_direct_q_clips_its_bootstrap_target_by_default() -> None:
    """Independent per-head targets are M unclipped Q-learners, not an ensemble."""
    from config.schema import DirectQCfg

    cfg = DirectQCfg()
    assert cfg.target_reduction == "min"
    assert cfg.target_subset_size == 2
    assert cfg.control_reduction == "min"
    with pytest.raises(ValueError, match="target_reduction"):
        DirectQCfg(target_reduction="mean")
    with pytest.raises(ValueError, match="target_subset_size"):
        DirectQCfg(num_heads=5, target_subset_size=1)
    with pytest.raises(ValueError, match="target_subset_size"):
        DirectQCfg(num_heads=3, target_subset_size=4)


def test_bootstrap_value_takes_a_subset_minimum() -> None:
    """The clipped target is a minimum over a fresh random subset of heads."""
    from algorithms.off_policy.values.backends import DirectActionValue

    backend = DirectActionValue.__new__(DirectActionValue)
    backend.num_heads = 4
    backend.target_subset_size = 2
    backend.subset_generator = torch.Generator(device="cpu").manual_seed(0)

    per_head = torch.tensor([[1.0, 9.0], [2.0, 8.0], [3.0, 7.0], [4.0, 6.0]])

    backend.target_reduction = "independent"
    for head_index in range(4):
        torch.testing.assert_close(
            backend._bootstrap_value(per_head, head_index), per_head[head_index]
        )

    backend.target_reduction = "min"
    for head_index in range(4):
        value = backend._bootstrap_value(per_head, head_index)
        assert value.shape == (2,)
        # A minimum over two of four heads: never above the pair-wise worst
        # case, never below the full ensemble minimum.
        assert torch.all(value >= per_head.min(dim=0).values)
        assert torch.all(value <= per_head.max(dim=0).values)

    backend.target_subset_size = 4
    torch.testing.assert_close(
        backend._bootstrap_value(per_head, 0), per_head.min(dim=0).values
    )


def test_clipped_target_never_exceeds_the_independent_one() -> None:
    """Whatever subset is drawn, clipping can only lower a head's target."""
    from algorithms.off_policy.values.backends import DirectActionValue

    backend = DirectActionValue.__new__(DirectActionValue)
    backend.num_heads = 3
    backend.target_subset_size = 2
    backend.target_reduction = "min"
    backend.subset_generator = torch.Generator(device="cpu").manual_seed(7)

    generator = torch.Generator().manual_seed(11)
    for _ in range(32):
        per_head = torch.randn(3, 8, generator=generator)
        clipped = torch.stack(
            [backend._bootstrap_value(per_head, head) for head in range(3)]
        )
        assert torch.all(clipped <= per_head.max(dim=0).values)


def test_bro_critic_is_layer_normalized_and_residual() -> None:
    from algorithms.off_policy.values.networks import BroNet, ScalarActionValue

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
    from algorithms.off_policy.values.networks import ScalarActionValue

    torch.manual_seed(0)
    critic = ScalarActionValue(3, 2, width=32, residual_blocks=2)
    observations = torch.randn(64, 3)
    actions = torch.randn(64, 2)

    on_distribution = critic(observations, actions).abs().max()
    far_away = critic(1_000.0 * observations, 1_000.0 * actions).abs().max()

    assert float(far_away) < 20.0 * float(on_distribution) + 10.0


def test_reset_reinitializes_every_head() -> None:
    from algorithms.off_policy.values.networks import ScalarActionValueEnsemble

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


def test_the_entropy_critic_exists_only_when_a_temperature_asks_for_it() -> None:
    """Successor features are reward-independent, so entropy needs its own critic."""
    from algorithms.off_policy.values.backends import SuccessorActionValue
    from config.schema import OffPolicyTrainCfg, SuccessorQCfg
    from reward.fixed import FixedLinearReward

    def build(use_entropy_critic: bool) -> SuccessorActionValue:
        return SuccessorActionValue(
            observation_dim=3,
            action_dim=2,
            feature_dim=2,
            reward=FixedLinearReward(
                ("alive", "effort"), torch.tensor([1.0, -0.1]), step_dt=0.02
            ),
            cfg=SuccessorQCfg(num_heads=2, width=16, residual_blocks=1, latent_dim=8),
            train_cfg=OffPolicyTrainCfg(),
            use_entropy_critic=use_entropy_critic,
            step_dt=0.02,
            device=torch.device("cpu"),
            seed=0,
        )

    assert build(False).entropy is None
    assert build(True).entropy is not None


def test_an_entropy_checkpoint_does_not_load_into_a_backend_without_one() -> None:
    from algorithms.off_policy.values.backends import SuccessorActionValue
    from config.schema import OffPolicyTrainCfg, SuccessorQCfg
    from reward.fixed import FixedLinearReward

    def build(use_entropy_critic: bool) -> SuccessorActionValue:
        return SuccessorActionValue(
            observation_dim=3,
            action_dim=2,
            feature_dim=2,
            reward=FixedLinearReward(
                ("alive", "effort"), torch.tensor([1.0, -0.1]), step_dt=0.02
            ),
            cfg=SuccessorQCfg(num_heads=2, width=16, residual_blocks=1, latent_dim=8),
            train_cfg=OffPolicyTrainCfg(),
            use_entropy_critic=use_entropy_critic,
            step_dt=0.02,
            device=torch.device("cpu"),
            seed=0,
        )

    with_entropy = build(True)
    with_entropy.fit_feature_scale(build_rollout(feature_dim=2))
    state = with_entropy.save_state()

    with pytest.raises(ValueError, match="entropy-critic configuration differs"):
        build(False).load_state(state, load_optimizer=True)


def test_an_optimistic_quantile_scores_above_the_median() -> None:
    """Optimism is the other half of the range: pulled toward disagreement."""
    from algorithms.off_policy.values.backends import _ensemble_quantile

    per_head = torch.tensor([[1.0, 5.0], [2.0, 6.0], [9.0, 7.0]])

    pessimistic = _ensemble_quantile(per_head, 0.0)
    median = _ensemble_quantile(per_head, 0.5)
    optimistic = _ensemble_quantile(per_head, 1.0)

    torch.testing.assert_close(pessimistic, torch.tensor([1.0, 5.0]))
    torch.testing.assert_close(median, torch.tensor([2.0, 6.0]))
    torch.testing.assert_close(optimistic, torch.tensor([9.0, 7.0]))
    assert bool((optimistic >= median).all()) and bool((median >= pessimistic).all())
