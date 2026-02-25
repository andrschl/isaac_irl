import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from algorithms.irl import IRL, IRLCfg
from reward_model.base import BaseRewardModel
from utils.runtime_context import RuntimeContext
from storage.feature_storage import FeatureBufCfg, FeatureTrajectoryBuffer


class LinearFeatureReward(BaseRewardModel):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        if weight.ndim != 1:
            raise ValueError(f"Expected weight shape [D], got {tuple(weight.shape)}")
        self.linear = nn.Linear(weight.shape[0], 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight.unsqueeze(0))

    @property
    def is_linear(self) -> bool:
        return True

    def get_reward_from_features(self, feats: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        del mask
        return self.linear(feats)


class NonLinearFeatureReward(BaseRewardModel):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.tensor(0.25))

    @property
    def is_linear(self) -> bool:
        return False

    def get_reward_from_features(self, feats: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        rewards = (feats * self.scale).sum(dim=-1) + self.bias
        if mask is not None:
            rewards = rewards * mask.to(dtype=rewards.dtype)
        return rewards


def _manual_discounted_returns(rewards: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Args:
        rewards: [B, T]
        mask: [B, T]
    Returns:
        returns: [B]
    """
    if rewards.shape != mask.shape:
        raise ValueError(f"Expected rewards/mask shape match, got {rewards.shape=} and {mask.shape=}")

    batch_size, traj_len = rewards.shape
    out = torch.zeros(batch_size, dtype=rewards.dtype, device=rewards.device)
    for time_idx in range(traj_len):
        out = out + (gamma**time_idx) * rewards[:, time_idx] * mask[:, time_idx].to(dtype=rewards.dtype)
    return out


def _make_buffer(gamma: float, feature_dim: int = 2) -> FeatureTrajectoryBuffer:
    ctx = RuntimeContext(num_envs=1, feature_dim=feature_dim, device="cpu")
    cfg = FeatureBufCfg(min_ep_len=1, store_dtype=torch.float32)
    return FeatureTrajectoryBuffer(cfg=cfg, ctx=ctx, gamma=gamma)


def test_eval_discounted_returns_from_model_non_linear_matches_manual_sum():
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

    returns = FeatureTrajectoryBuffer.eval_discounted_returns_from_model(
        feats=feats,
        mask=mask,
        reward_model=reward_model,
        gamma=gamma,
    )

    assert torch.allclose(returns, expected_returns)


def test_eval_discounted_returns_from_model_linear_matches_discounted_features():
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
    discounted_feats = FeatureTrajectoryBuffer.discounted_feature_returns(feats, mask, gamma)
    expected_returns = reward_model.get_reward_from_features(discounted_feats).squeeze(-1)

    returns = FeatureTrajectoryBuffer.eval_discounted_returns_from_model(
        feats=feats,
        mask=mask,
        reward_model=reward_model,
        gamma=gamma,
    )

    assert torch.allclose(returns, expected_returns)


def test_eval_discounted_returns_from_model_non_linear_aligns_compute_dtype():
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

    returns = FeatureTrajectoryBuffer.eval_discounted_returns_from_model(
        feats=feats,
        mask=mask,
        reward_model=reward_model,
        gamma=gamma,
    )

    assert returns.dtype == reward_param.dtype
    assert returns.device == reward_param.device
    assert torch.allclose(returns, expected_returns)


def test_eval_discounted_returns_from_model_linear_aligns_compute_dtype():
    gamma = 0.9
    feats = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 1.0], [4.0, 3.0]],
            [[1.0, 1.0], [0.0, 2.0], [5.0, 8.0]],
        ],
        dtype=torch.float16,
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])

    reward_model = LinearFeatureReward(weight=torch.tensor([2.0, -1.0], dtype=torch.float32))
    reward_param = next(reward_model.parameters())
    discounted_feats = FeatureTrajectoryBuffer.discounted_feature_returns(feats, mask, gamma)
    discounted_feats = discounted_feats.to(device=reward_param.device, dtype=reward_param.dtype)
    expected_returns = reward_model.get_reward_from_features(discounted_feats).squeeze(-1)

    returns = FeatureTrajectoryBuffer.eval_discounted_returns_from_model(
        feats=feats,
        mask=mask,
        reward_model=reward_model,
        gamma=gamma,
    )

    assert returns.dtype == reward_param.dtype
    assert returns.device == reward_param.device
    assert torch.allclose(returns, expected_returns)


@pytest.mark.parametrize("reward_kind", ["linear", "non_linear"])
def test_sample_and_eval_returns_output_contract(reward_kind: str):
    gamma = 0.95
    feature_dim = 2
    reward_model: BaseRewardModel
    if reward_kind == "linear":
        reward_model = LinearFeatureReward(weight=torch.tensor([1.0, 2.0]))
    else:
        reward_model = NonLinearFeatureReward(feature_dim=feature_dim)

    buffer = _make_buffer(gamma=gamma, feature_dim=feature_dim)
    buffer.add_episode(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))

    out = buffer.sample_and_eval_returns(reward_model=reward_model, batch_size=1, device="cpu")

    assert set(out.keys()) == {"feats", "mask", "lengths", "returns"}
    assert out["feats"].shape == (1, 2, feature_dim)
    assert out["mask"].shape == (1, 2)
    assert out["lengths"].shape == (1,)
    assert out["returns"].shape == (1,)


class _DummyRlAlg:
    learning_rate = 1e-3


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

    irl = IRL(
        rl_alg=_DummyRlAlg(),
        reward=reward_model,
        gamma=gamma,
        cfg=IRLCfg(batch_size=2),
        device="cpu",
    )

    returns = FeatureTrajectoryBuffer.eval_discounted_returns_from_model(
        feats=feats,
        mask=mask,
        reward_model=reward_model,
        gamma=gamma,
    )
    lengths = mask.to(dtype=returns.dtype).sum(dim=1).clamp_min(1.0)
    expected = (returns / lengths).mean()

    got = irl._eval_expected_return(feats, mask)
    assert torch.allclose(got, expected)


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

    irl = IRL(
        rl_alg=_DummyRlAlg(),
        reward=reward_model,
        gamma=gamma,
        cfg=IRLCfg(batch_size=2, normalize_returns_by_episode_length=False),
        device="cpu",
    )

    expected = FeatureTrajectoryBuffer.eval_discounted_returns_from_model(
        feats=feats,
        mask=mask,
        reward_model=reward_model,
        gamma=gamma,
    ).mean()

    got = irl._eval_expected_return(feats, mask)
    assert torch.allclose(got, expected)


def _make_irl_for_discount_cfg(
    *,
    gamma: float,
    discount_gamma: float | None,
    normalize_returns_by_episode_length: bool,
) -> IRL:
    reward_model = LinearFeatureReward(weight=torch.tensor([1.0, 0.0], dtype=torch.float32))
    irl = IRL(
        rl_alg=_DummyRlAlg(),
        reward=reward_model,
        gamma=gamma,
        cfg=IRLCfg(
            batch_size=1,
            num_learning_epochs=1,
            reward_loss_coef=1.0,
            max_grad_norm=10.0,
            discount_gamma=discount_gamma,
            normalize_returns_by_episode_length=normalize_returns_by_episode_length,
        ),
        device="cpu",
    )

    runtime_ctx = RuntimeContext(num_envs=1, feature_dim=2, device="cpu")
    buffer_cfg = FeatureBufCfg(min_ep_len=1, store_dtype=torch.float32)
    irl.init_expert_storage(runtime_ctx=runtime_ctx, cfg=buffer_cfg, num_envs=1)
    irl.init_imitator_storage(runtime_ctx=runtime_ctx, cfg=buffer_cfg)

    expert_episode = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    imitator_episode = torch.tensor([[2.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    irl.add_expert_episode(expert_episode)
    if irl.imitator_storage is None:
        raise RuntimeError("Imitator storage initialization failed in test.")
    irl.imitator_storage.add_episode(imitator_episode)
    return irl


def test_irl_reward_update_normalizes_returns_by_episode_length_by_default():
    gamma = 0.9
    irl = _make_irl_for_discount_cfg(
        gamma=gamma,
        discount_gamma=None,
        normalize_returns_by_episode_length=True,
    )

    reward_loss, grad_norm = irl.reward_update()

    assert reward_loss == pytest.approx(0.95, abs=1e-5)
    assert grad_norm > 0.0


def test_irl_reward_update_uses_separate_discount_gamma_when_configured():
    gamma = 0.9
    irl = _make_irl_for_discount_cfg(
        gamma=gamma,
        discount_gamma=0.5,
        normalize_returns_by_episode_length=False,
    )

    reward_loss, grad_norm = irl.reward_update()

    assert reward_loss == pytest.approx(1.5, abs=1e-5)
    assert grad_norm > 0.0


def test_irl_rejects_invalid_discount_gamma():
    with pytest.raises(ValueError, match="discount_gamma"):
        _ = IRL(
            rl_alg=_DummyRlAlg(),
            reward=LinearFeatureReward(weight=torch.tensor([1.0, 0.0], dtype=torch.float32)),
            gamma=0.9,
            cfg=IRLCfg(discount_gamma=0.0),
            device="cpu",
        )
