"""Mixtures act on the shared reward basis, never on successor TD targets."""

from types import SimpleNamespace

import pytest
import torch

from config.schema import RewardCfg, RewardModelCfg, TrainCfg
from interfaces import FeatureSpec
from reward.fixed import FixedLinearReward
from reward.model import LinearFeatureRewardModel
from reward.scorer import AdditiveLinearRewardScorer, LearnedLinearRewardScorer
from training.build import (
    build_reward_model,
    build_reward_scorer,
    learned_feature_columns,
)


@pytest.mark.parametrize("coefficients", [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)])
def test_mixture_preserves_units_and_live_signed_weights(coefficients):
    model = LinearFeatureRewardModel(RewardModelCfg(is_linear=True), num_features=1)
    fixed = FixedLinearReward(
        ("terminal_success", "reach"), torch.tensor([1.0, 0.0]), step_dt=0.02
    )
    scorer = AdditiveLinearRewardScorer(
        fixed,
        LearnedLinearRewardScorer(model, feature_names=("reach",), step_dt=0.02),
        fixed_coefficient=coefficients[0],
        learned_coefficient=coefficients[1],
    )
    features = torch.tensor([[[50.0, 2.0], [0.0, 3.0]]])  # [B,T,D]
    for learned_weight in (-2.0, 3.0):
        with torch.no_grad():
            model.reward.weight.fill_(learned_weight)
        expected = (
            coefficients[0] * fixed.score(features)
            + coefficients[1] * 0.02 * learned_weight * features[..., 1]
        )
        torch.testing.assert_close(scorer.score(features), expected)
        torch.testing.assert_close(fixed.weights, torch.tensor([1.0, 0.0]))


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf")])
@pytest.mark.parametrize("key", ["environment_weight", "learned_weight"])
def test_invalid_mixture_weights_rejected(key, bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        RewardCfg(source="environment_plus_learned", **{key: bad})


def test_zero_mixture_and_ignored_coefficients_rejected():
    with pytest.raises(ValueError, match="positive"):
        RewardCfg(
            source="environment_plus_learned", environment_weight=0, learned_weight=0
        )
    with pytest.raises(ValueError, match="require source"):
        RewardCfg(source="environment", learned_weight=0.5)


def test_builder_passes_mixture_coefficients(monkeypatch):
    monkeypatch.setattr(
        "training.build.extract_ground_truth_reward_weights",
        lambda env: {"a": 2.0},
    )
    cfg = TrainCfg(
        device="cpu",
        reward=RewardCfg(
            source="environment_plus_learned",
            environment_weight=0.25,
            learned_weight=0.75,
            model=RewardModelCfg(is_linear=True),
        ),
    )
    model = LinearFeatureRewardModel(cfg.reward.model, num_features=1)
    with torch.no_grad():
        model.reward.weight.fill_(-4)
    scorer = build_reward_scorer(
        cfg,
        env=SimpleNamespace(unwrapped=SimpleNamespace(step_dt=0.02)),
        feature_source=SimpleNamespace(spec=FeatureSpec(dim=1, names=("a",))),
        reward_model=model,
    )
    torch.testing.assert_close(scorer.weights, torch.tensor([-2.5]))


def test_named_scaled_basis_keeps_raw_collection_and_orders_model_scales():
    cfg = TrainCfg(
        device="cpu",
        reward=RewardCfg(
            source="learned",
            model=RewardModelCfg(is_linear=True),
            learned_feature_scales={"terminal_success": 1.0, "reach": 5.0},
        ),
    )
    source = SimpleNamespace(
        spec=FeatureSpec(
            dim=4, names=("reach", "joint_vel", "terminal_success", "action_rate")
        )
    )
    columns = learned_feature_columns(cfg, source)
    assert columns.names == ("reach", "terminal_success")
    torch.testing.assert_close(
        columns.apply(torch.tensor([[2.0, 9.0, 50.0, 3.0]])),
        torch.tensor([[2.0, 50.0]]),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(step_dt=0.02))
    model = build_reward_model(cfg, env=env, feature_source=source, device="cpu")
    torch.testing.assert_close(model.feature_scales, torch.tensor([5.0, 1.0]))
    scorer = build_reward_scorer(
        cfg, env=env, feature_source=source, reward_model=model
    )
    with torch.no_grad():
        model.reward.weight.copy_(torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(scorer.weights, torch.tensor([0.4, 0.0, 3.0, 0.0]))
    assert scorer.score(torch.tensor([[2.0, 9.0, 50.0, 3.0]])).item() == pytest.approx(
        3.016
    )


def test_environment_source_builds_no_reward_learner_or_scorer():
    cfg = TrainCfg(reward=RewardCfg(source="environment"))
    source = SimpleNamespace(spec=FeatureSpec(dim=1, names=("terminal_success",)))
    env = SimpleNamespace(unwrapped=SimpleNamespace(step_dt=0.02))
    assert build_reward_model(cfg, env=env, feature_source=source, device="cpu") is None
    assert (
        build_reward_scorer(cfg, env=env, feature_source=source, reward_model=None)
        is None
    )


def test_named_scaled_basis_rejects_unknown_features():
    cfg = TrainCfg(
        reward=RewardCfg(
            source="learned",
            model=RewardModelCfg(is_linear=True),
            learned_feature_scales={"missing": 1.0},
        )
    )
    source = SimpleNamespace(spec=FeatureSpec(dim=1, names=("reach",)))
    with pytest.raises(ValueError, match="not collected"):
        learned_feature_columns(cfg, source)
