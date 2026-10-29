# ruff: noqa: E402
"""The reward model's data contract: features in, per-step rewards out."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from config.schema import RewardModelCfg
from reward import DenseFeatureRewardModel, LinearFeatureRewardModel


def _linear(num_features: int = 3, **cfg_overrides) -> LinearFeatureRewardModel:
    return LinearFeatureRewardModel(
        RewardModelCfg(is_linear=True, **cfg_overrides), num_features=num_features
    )


def _dense(num_features: int = 3, **cfg_overrides) -> DenseFeatureRewardModel:
    settings = {"is_linear": False, "hidden_dims": (8,)}
    settings.update(cfg_overrides)
    return DenseFeatureRewardModel(
        RewardModelCfg(**settings), num_features=num_features
    )


def test_a_batch_of_steps_scores_one_reward_each() -> None:
    rewards = _dense().get_reward_from_features(torch.zeros(5, 3))

    assert rewards.shape == (5,)


def test_a_batch_of_trajectories_keeps_its_time_axis() -> None:
    rewards = _dense().get_reward_from_features(torch.zeros(2, 4, 3))

    assert rewards.shape == (2, 4)


def test_masked_steps_contribute_nothing() -> None:
    model = _dense()
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    rewards = model.get_reward_from_features(torch.ones(2, 4, 3), mask)

    assert torch.equal(rewards == 0, ~mask)


def test_a_wrong_feature_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match="Expected feature dimension D=3"):
        _dense().get_reward_from_features(torch.zeros(5, 4))


def test_a_linear_reward_has_no_bias() -> None:
    """A constant shift in reward changes nothing, so it must not be learnable."""
    assert _linear().reward.bias is None


def test_named_weights_land_in_feature_order() -> None:
    model = _linear(num_features=3)

    model.set_named_weights(
        ("reach", "lift", "goal"), {"lift": 2.0, "reach": -1.0, "goal": 0.0}
    )

    torch.testing.assert_close(
        model.reward.weight.detach().squeeze(0), torch.tensor([-1.0, 2.0, 0.0])
    )


def test_a_feature_with_no_weight_is_refused_rather_than_zeroed() -> None:
    """A silently zeroed term would look like a reward the env does not have."""
    model = _linear(num_features=2)

    with pytest.raises(ValueError, match="missing feature names"):
        model.set_named_weights(("reach", "lift"), {"reach": 1.0})


def test_projection_pulls_weights_back_onto_the_l2_ball() -> None:
    model = _linear(
        num_features=2, linear_projection="l2_ball", linear_projection_radius=1.0
    )
    with torch.no_grad():
        model.reward.weight.copy_(torch.tensor([[3.0, 4.0]]))

    model.project_weights()

    assert float(model.reward.weight.detach().norm().item()) == pytest.approx(1.0)


def test_l2_regularization_grows_with_the_weights() -> None:
    model = _linear(num_features=2, regularization="l2", regularization_strength=0.5)
    with torch.no_grad():
        model.reward.weight.copy_(torch.tensor([[1.0, 1.0]]))

    assert float(model.get_regularization_loss().item()) == pytest.approx(1.0)


def test_no_regularization_costs_nothing() -> None:
    assert float(_linear(regularization="none").get_regularization_loss().item()) == 0.0


def test_each_model_rejects_the_other_one_s_config() -> None:
    with pytest.raises(ValueError, match="is_linear"):
        DenseFeatureRewardModel(RewardModelCfg(is_linear=True), num_features=3)
    with pytest.raises(ValueError, match="is_linear"):
        LinearFeatureRewardModel(RewardModelCfg(is_linear=False), num_features=3)


def test_a_dense_model_cannot_take_a_linear_projection() -> None:
    with pytest.raises(ValueError, match="linear_projection"):
        _dense(linear_projection="l2_ball")


def test_a_dense_model_needs_hidden_layers() -> None:
    with pytest.raises(ValueError, match="hidden_dims"):
        _dense(hidden_dims=())


def test_a_reward_model_needs_a_positive_feature_dimension() -> None:
    with pytest.raises(ValueError, match="num_features"):
        _linear(num_features=0)
