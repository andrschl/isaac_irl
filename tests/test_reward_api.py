# ruff: noqa: E402

import pytest


torch = pytest.importorskip("torch")

from reward_model import (
    DenseFeatureRewardModel,
    LinearFeatureRewardModel,
    RewardModelCfg,
)


def test_reward_model_feature_api_output_shape_2d():
    model = DenseFeatureRewardModel(
        RewardModelCfg(
            num_features=4,
            hidden_dims=(16, 8),
            is_linear=False,
            activation="relu",
        )
    )
    feats = torch.randn(5, 4)
    out = model.get_reward_from_features(feats)
    assert out.shape == (5,)


def test_reward_model_feature_api_output_shape_3d_and_masking():
    model = DenseFeatureRewardModel(
        RewardModelCfg(
            num_features=4,
            hidden_dims=(8,),
            is_linear=False,
            activation="elu",
        )
    )
    feats = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool)
    out = model.get_reward_from_features(feats, mask=mask)
    assert out.shape == (2, 3)
    assert torch.all(out[~mask] == 0.0)


def test_linear_reward_model_is_bias_free():
    model = LinearFeatureRewardModel(
        RewardModelCfg(num_features=3, is_linear=True, hidden_dims=(8,))
    )
    assert model.reward.bias is None


def test_linear_reward_model_sets_weights_in_feature_name_order():
    model = LinearFeatureRewardModel(RewardModelCfg(num_features=3, is_linear=True))
    model.set_named_weights(
        ["velocity", "height", "effort"],
        {"height": 15.0, "effort": -0.001, "velocity": 2.0},
    )
    torch.testing.assert_close(
        model.reward.weight,
        torch.tensor([[2.0, 15.0, -0.001]]),
    )


def test_linear_reward_model_rejects_missing_named_weight():
    model = LinearFeatureRewardModel(RewardModelCfg(num_features=2, is_linear=True))
    with pytest.raises(ValueError, match="missing feature names"):
        model.set_named_weights(["height", "effort"], {"height": 15.0})


def test_linear_reward_model_l2_ball_projection():
    cfg = RewardModelCfg(
        num_features=4,
        is_linear=True,
        linear_projection="l2_ball",
        linear_projection_radius=1.0,
    )
    model = LinearFeatureRewardModel(cfg)
    model.reward.weight.data.fill_(2.0)
    model.project_weights()
    assert model.reward.weight.data.norm(p=2).item() <= 1.0 + 1e-5


def test_linear_reward_model_l1_ball_projection():
    cfg = RewardModelCfg(
        num_features=4,
        is_linear=True,
        linear_projection="l1_ball",
        linear_projection_radius=1.0,
    )
    model = LinearFeatureRewardModel(cfg)
    model.reward.weight.data.fill_(1.0)
    model.project_weights()
    assert model.reward.weight.data.abs().sum().item() <= 1.0 + 1e-5


def test_reward_model_l2_regularization_loss():
    cfg = RewardModelCfg(
        num_features=4,
        is_linear=False,
        hidden_dims=(8,),
        regularization="l2",
        regularization_strength=0.1,
    )
    model = DenseFeatureRewardModel(cfg)
    reg = model.get_regularization_loss()
    assert reg.ndim == 0
    assert reg.item() > 0


def test_reward_model_regularization_none_returns_zero():
    cfg = RewardModelCfg(num_features=4, is_linear=True, regularization="none")
    model = LinearFeatureRewardModel(cfg)
    reg = model.get_regularization_loss()
    assert reg.item() == 0.0


def test_reward_model_apply_proximal_step_l1():
    cfg = RewardModelCfg(
        num_features=4, is_linear=True, regularization="l1", regularization_strength=0.5
    )
    model = LinearFeatureRewardModel(cfg)
    model.reward.weight.data.fill_(2.0)
    model.apply_proximal_step()
    expected = 1.5
    torch.testing.assert_close(
        model.reward.weight.data, torch.full_like(model.reward.weight.data, expected)
    )


def test_dense_reward_model_rejects_is_linear_true():
    with pytest.raises(ValueError, match="is_linear"):
        DenseFeatureRewardModel(RewardModelCfg(num_features=2, is_linear=True))


def test_linear_reward_model_rejects_is_linear_false():
    with pytest.raises(ValueError, match="is_linear"):
        LinearFeatureRewardModel(RewardModelCfg(num_features=2, is_linear=False))


def test_dense_reward_model_rejects_linear_projection():
    with pytest.raises(ValueError, match="LinearFeatureRewardModel"):
        DenseFeatureRewardModel(
            RewardModelCfg(num_features=2, is_linear=False, linear_projection="l2_ball")
        )


def test_dense_reward_model_rejects_empty_hidden_dims():
    with pytest.raises(ValueError, match="hidden_dims"):
        DenseFeatureRewardModel(
            RewardModelCfg(num_features=2, is_linear=False, hidden_dims=())
        )
