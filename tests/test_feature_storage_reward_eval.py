# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from algorithms.irl import IrlAlgorithm, IRLCfg
from config.schema import RewardModelCfg
from interfaces import FeatureSpec, RuntimeContext
from reward import LinearFeatureRewardModel
from reward.base import BaseRewardModel
from storage.feature_storage import FeatureBufCfg, FeatureTrajectoryBuffer


class LinearFeatureReward(LinearFeatureRewardModel):
    def __init__(self, weight: torch.Tensor) -> None:
        if weight.ndim != 1:
            raise ValueError(f"Expected weight shape [D], got {tuple(weight.shape)}")
        super().__init__(
            RewardModelCfg(is_linear=True), num_features=int(weight.shape[0])
        )
        with torch.no_grad():
            self.reward.weight.copy_(weight.unsqueeze(0))


class NonLinearFeatureReward(BaseRewardModel):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.tensor(0.25))

    def get_reward_from_features(
        self, feats: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        rewards = (feats * self.scale).sum(dim=-1) + self.bias
        if mask is not None:
            rewards = rewards * mask.to(dtype=rewards.dtype)
        return rewards


def _manual_discounted_returns(
    rewards: torch.Tensor, mask: torch.Tensor, gamma: float
) -> torch.Tensor:
    """
    Args:
        rewards: [B, T]
        mask: [B, T]
    Returns:
        returns: [B]
    """
    if rewards.shape != mask.shape:
        raise ValueError(
            f"Expected rewards/mask shape match, got {rewards.shape=} and {mask.shape=}"
        )

    batch_size, traj_len = rewards.shape
    out = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)
    for time_idx in range(traj_len):
        out = out + (gamma**time_idx) * rewards[:, time_idx] * mask[:, time_idx].to(
            dtype=rewards.dtype
        )
    return out


def _make_buffer(gamma: float, feature_dim: int = 2) -> FeatureTrajectoryBuffer:
    ctx = RuntimeContext(num_envs=1, feature_dim=feature_dim, device="cpu")
    cfg = FeatureBufCfg(min_ep_len=1, store_dtype=torch.float32)
    return FeatureTrajectoryBuffer(cfg=cfg, ctx=ctx, gamma=gamma)


def test_discounted_returns_from_features_non_linear_matches_manual_sum():
    gamma = 0.6
    feats = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
            [[13.0, 17.0], [19.0, 23.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, True], [True, True, False]])

    reward_model = NonLinearFeatureReward(feature_dim=2)
    with torch.no_grad():
        reward_model.scale.copy_(torch.tensor([0.5, -0.25]))
        reward_model.bias.copy_(torch.tensor(1.5))

    rewards = reward_model.get_reward_from_features(feats, mask)
    expected_returns = _manual_discounted_returns(rewards, mask, gamma)

    returns = reward_model.discounted_returns_from_features(
        feats=feats,
        mask=mask,
        gamma=gamma,
    )

    assert torch.allclose(returns, expected_returns)


def test_discounted_returns_from_features_linear_matches_discounted_features():
    gamma = 0.9
    feats = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0], [4.0, 3.0]],
            [[1.0, 1.0], [0.0, 2.0], [5.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])

    reward_model = LinearFeatureReward(weight=torch.tensor([2.0, -1.0]))
    discounted_feats = FeatureTrajectoryBuffer.discounted_feature_returns(
        feats, mask, gamma
    )
    expected_returns = reward_model.get_reward_from_features(discounted_feats).squeeze(
        -1
    )

    returns = reward_model.discounted_returns_from_features(
        feats=feats,
        mask=mask,
        gamma=gamma,
    )

    assert torch.allclose(returns, expected_returns)


def test_discounted_returns_from_features_non_linear_aligns_compute_dtype():
    gamma = 0.6
    feats = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
            [[13.0, 17.0], [19.0, 23.0], [0.0, 0.0]],
        ],
        dtype=torch.float16,
    )
    mask = torch.tensor([[True, True, True], [True, True, False]])

    reward_model = NonLinearFeatureReward(feature_dim=2)
    with torch.no_grad():
        reward_model.scale.copy_(torch.tensor([0.5, -0.25]))
        reward_model.bias.copy_(torch.tensor(1.5))

    reward_param = next(reward_model.parameters())
    feats_compute = feats.to(device=reward_param.device, dtype=reward_param.dtype)
    mask_compute = mask.to(device=reward_param.device, dtype=torch.bool)
    rewards = reward_model.get_reward_from_features(feats_compute, mask_compute)
    expected_returns = _manual_discounted_returns(rewards, mask_compute, gamma)

    returns = reward_model.discounted_returns_from_features(
        feats=feats,
        mask=mask,
        gamma=gamma,
    )

    assert returns.dtype == reward_param.dtype
    assert returns.device == reward_param.device
    assert torch.allclose(returns, expected_returns)


def test_discounted_returns_from_features_linear_aligns_compute_dtype():
    gamma = 0.9
    feats = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0], [4.0, 3.0]],
            [[1.0, 1.0], [0.0, 2.0], [5.0, 8.0]],
        ],
        dtype=torch.float16,
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])

    reward_model = LinearFeatureReward(
        weight=torch.tensor([2.0, -1.0], dtype=torch.float32)
    )
    reward_param = next(reward_model.parameters())
    feats_compute = feats.to(device=reward_param.device, dtype=reward_param.dtype)
    mask_compute = mask.to(device=reward_param.device, dtype=torch.bool)
    discounted_feats = FeatureTrajectoryBuffer.discounted_feature_returns(
        feats_compute, mask_compute, gamma
    )
    expected_returns = reward_model.get_reward_from_features(discounted_feats).squeeze(
        -1
    )

    returns = reward_model.discounted_returns_from_features(
        feats=feats,
        mask=mask,
        gamma=gamma,
    )

    assert returns.dtype == reward_param.dtype
    assert returns.device == reward_param.device
    assert torch.allclose(returns, expected_returns)


def test_irl_eval_expected_return_matches_shared_helper():
    gamma = 0.8
    feats = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[2.0, 1.0], [1.0, 0.5], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, False], [True, True, False]])
    reward_model = NonLinearFeatureReward(feature_dim=2)

    irl = IrlAlgorithm(
        reward=reward_model,
        feature_spec=FeatureSpec(dim=2),
        gamma=gamma,
        cfg=IRLCfg(batch_size=2),
        device="cpu",
    )

    returns = reward_model.discounted_returns_from_features(
        feats=feats,
        mask=mask,
        gamma=gamma,
    )
    lengths = mask.to(dtype=returns.dtype).sum(dim=1).clamp_min(1.0)
    expected_scalar = (returns / lengths).mean()
    discounted_feats = FeatureTrajectoryBuffer.discounted_feature_returns(
        feats, mask, gamma
    )
    expected_per_feature = (discounted_feats / lengths.unsqueeze(1)).mean(dim=0)

    got_scalar, got_per_feature = irl.eval_expected_return(feats, mask)
    assert torch.allclose(got_scalar, expected_scalar)
    assert torch.allclose(got_per_feature, expected_per_feature)


def test_irl_eval_expected_return_can_disable_length_normalization():
    gamma = 0.8
    feats = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
            [[2.0, 1.0], [1.0, 0.5], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([[True, True, False], [True, True, False]])
    reward_model = NonLinearFeatureReward(feature_dim=2)

    irl = IrlAlgorithm(
        reward=reward_model,
        feature_spec=FeatureSpec(dim=2),
        gamma=gamma,
        cfg=IRLCfg(batch_size=2, normalize_returns_by_episode_length=False),
        device="cpu",
    )

    expected_scalar = reward_model.discounted_returns_from_features(
        feats=feats,
        mask=mask,
        gamma=gamma,
    ).mean()
    expected_per_feature = FeatureTrajectoryBuffer.discounted_feature_returns(
        feats,
        mask,
        gamma,
    ).mean(dim=0)

    got_scalar, got_per_feature = irl.eval_expected_return(feats, mask)
    assert torch.allclose(got_scalar, expected_scalar)
    assert torch.allclose(got_per_feature, expected_per_feature)


def _make_irl_for_discount_cfg(
    *,
    gamma: float,
    discount_gamma: float | None,
    normalize_returns_by_episode_length: bool,
) -> IrlAlgorithm:
    reward_model = LinearFeatureReward(
        weight=torch.tensor([1.0, 0.0], dtype=torch.float32)
    )
    buffer_cfg = FeatureBufCfg(min_ep_len=1, store_dtype=torch.float32)
    irl = IrlAlgorithm(
        reward=reward_model,
        feature_spec=FeatureSpec(dim=2),
        gamma=gamma,
        cfg=IRLCfg(
            batch_size=1,
            gradient_steps=1,
            reward_loss_coef=1.0,
            max_grad_norm=10.0,
            discount_gamma=discount_gamma,
            normalize_returns_by_episode_length=normalize_returns_by_episode_length,
        ),
        device="cpu",
        imitator_buffer_cfg=buffer_cfg,
        expert_buffer_cfg=buffer_cfg,
    )

    expert_episode = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    imitator_episode = torch.tensor([[2.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    irl.add_expert_episode(expert_episode)
    irl.imitator_storage.append_episode(imitator_episode)
    return irl


def test_irl_reward_update_normalizes_returns_by_episode_length_by_default():
    gamma = 0.9
    irl = _make_irl_for_discount_cfg(
        gamma=gamma,
        discount_gamma=None,
        normalize_returns_by_episode_length=True,
    )

    metrics = irl.update()

    assert metrics["IRL/reward_loss"] == pytest.approx(0.95, abs=1e-5)
    assert metrics["IRL/feature_exp_diff_norm"] > 0.0


def test_irl_reward_update_uses_separate_discount_gamma_when_configured():
    gamma = 0.9
    irl = _make_irl_for_discount_cfg(
        gamma=gamma,
        discount_gamma=0.5,
        normalize_returns_by_episode_length=False,
    )

    metrics = irl.update()

    assert metrics["IRL/reward_loss"] == pytest.approx(1.5, abs=1e-5)
    assert metrics["IRL/feature_exp_diff_norm"] > 0.0


def test_irl_rejects_invalid_discount_gamma():
    with pytest.raises(ValueError, match="discount_gamma"):
        _ = IrlAlgorithm(
            reward=LinearFeatureReward(
                weight=torch.tensor([1.0, 0.0], dtype=torch.float32)
            ),
            feature_spec=FeatureSpec(dim=2),
            gamma=0.9,
            cfg=IRLCfg(discount_gamma=0.0),
            device="cpu",
        )


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"gradient_steps": 0}, "gradient_steps"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"max_grad_norm": 0.0}, "max_grad_norm"),
        ({"reward_loss_coef": -1.0}, "reward_loss_coef"),
        ({"reward_learning_rate": 0.0}, "reward_learning_rate"),
        ({"discount_gamma": 0.0}, "discount_gamma"),
    ],
)
def test_irl_cfg_rejects_invalid_values(overrides: dict, match: str):
    """A config validates itself, so an invalid one cannot reach an algorithm."""
    with pytest.raises(ValueError, match=match):
        IRLCfg(**overrides)


def test_reward_weight_cosine_reports_alignment_with_the_environment_reward():
    """The direct read on whether IRL is recovering the task's reward."""
    reward_model = LinearFeatureRewardModel(
        RewardModelCfg(is_linear=True), num_features=2
    )
    irl = IrlAlgorithm(
        reward=reward_model,
        feature_spec=FeatureSpec(dim=2, names=("reach", "lift")),
        gamma=0.99,
        cfg=IRLCfg(),
        device="cpu",
        environment_reward_weights=torch.tensor([1.0, 0.0]),
    )

    with torch.no_grad():
        reward_model.reward.weight.copy_(torch.tensor([[2.0, 0.0]]))
    assert irl.reward_weight_cosine() == pytest.approx(1.0)

    with torch.no_grad():
        reward_model.reward.weight.copy_(torch.tensor([[0.0, 1.0]]))
    assert irl.reward_weight_cosine() == pytest.approx(0.0, abs=1e-6)

    assert "IRL/reward_weight_cosine" in irl.train_metrics()


def test_reward_weight_cosine_is_absent_when_the_task_exposes_no_weights():
    irl = IrlAlgorithm(
        reward=LinearFeatureRewardModel(RewardModelCfg(is_linear=True), num_features=2),
        feature_spec=FeatureSpec(dim=2),
        gamma=0.99,
        cfg=IRLCfg(),
        device="cpu",
    )

    assert irl.reward_weight_cosine() is None
    assert "IRL/reward_weight_cosine" not in irl.train_metrics()
