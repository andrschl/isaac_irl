import pytest


torch = pytest.importorskip("torch")

from reward_model import RewardModel, RewardModelCfg


def test_reward_model_feature_api_output_shape_2d():
    model = RewardModel(
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
    model = RewardModel(
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
    model = RewardModel(RewardModelCfg(num_features=3, is_linear=True, hidden_dims=(8,)))
    assert model.is_linear is True
    assert model.reward.bias is None
