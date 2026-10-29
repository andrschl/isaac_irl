"""Exact linear IRL uses every complete episode in the configured feature units."""

import math

import pytest
import torch

from algorithms.irl import IrlAlgorithm
from config.schema import IRLCfg, RewardModelCfg
from interfaces import FeatureSpec
from reward.model import LinearFeatureRewardModel
from reward.scorer import LearnedLinearRewardScorer


def _learner(*, regularization_strength: float = 1e-4) -> IrlAlgorithm:
    model = LinearFeatureRewardModel(
        RewardModelCfg(
            is_linear=True,
            regularization="l2",
            regularization_strength=regularization_strength,
            linear_projection="l2_ball",
            linear_projection_radius=1.0,
        ),
        num_features=2,
        feature_scales=(2.0, 1.0),
    )
    return IrlAlgorithm(
        model,
        feature_spec=FeatureSpec(dim=2, names=("goal", "terminal_success")),
        gamma=1.0,
        step_dt=1.0,
        cfg=IRLCfg(
            solver="exact_linear",
            gradient_steps=0,
            weight_decay=0,
            normalize_returns_by_episode_length=False,
        ),
    )


def test_exact_solve_uses_every_episode_and_respects_feature_scales() -> None:
    learner = _learner()
    learner.add_expert_episode(torch.tensor([[2.0, 0.0]]))
    learner.add_expert_episode(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
    learner.replace_imitator_episodes(
        torch.zeros(2, 2, 2), torch.tensor([[True, False], [True, True]])
    )

    metrics = learner.update()

    # Expert mean raw return [1,1]; scaled gap [0.5,1].
    expected = torch.tensor([0.5, 1.0]) / math.sqrt(1.25)
    torch.testing.assert_close(learner.reward_model.reward.weight.squeeze(0), expected)
    assert metrics["IRL/feature_exp_diff_norm"] == pytest.approx(math.sqrt(1.25))
    assert metrics["IRL/raw_feature_exp_diff_norm"] == pytest.approx(math.sqrt(2.0))
    assert metrics["IRL/reward_optimizer_steps"] == 0.0
    assert metrics["IRL/reward_exact_solves"] == 1.0
    assert metrics["IRL/reward_contribution/terminal_success"] == 0.0
    assert learner.reward_optimizer is None


def test_exact_solve_handles_interior_optimum_and_zero_gap() -> None:
    learner = _learner(regularization_strength=2.0)
    learner.add_expert_episode(torch.tensor([[2.0, 0.0]]))
    learner.replace_imitator_episodes(
        torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool)
    )
    learner.update()
    torch.testing.assert_close(
        learner.reward_model.reward.weight.squeeze(0), torch.tensor([0.25, 0.0])
    )
    learner.replace_imitator_episodes(
        torch.tensor([[[2.0, 0.0]]]), torch.ones(1, 1, dtype=torch.bool)
    )
    learner.update()
    torch.testing.assert_close(learner.reward_model.reward.weight, torch.zeros(1, 2))


def test_success_pulse_integrates_once_with_same_scorer_units() -> None:
    model = LinearFeatureRewardModel(
        RewardModelCfg(is_linear=True), num_features=2, feature_scales=(5.0, 1.0)
    )
    with torch.no_grad():
        model.reward.weight.copy_(torch.tensor([[2.0, 3.0]]))
    scorer = LearnedLinearRewardScorer(
        model, feature_names=("goal", "terminal_success"), step_dt=0.02
    )
    features = torch.tensor([[[1.0, 0.0], [1.0, 50.0]]])
    torch.testing.assert_close(scorer.weights, torch.tensor([0.4, 3.0]))
    torch.testing.assert_close(scorer.score(features), torch.tensor([[0.008, 3.008]]))
    torch.testing.assert_close(
        0.02 * model.get_reward_from_features(features), scorer.score(features)
    )


def test_exact_checkpoint_has_no_optimizer_and_restores_weights() -> None:
    learner = _learner()
    state = learner.save_state()
    assert "reward_optimizer_state_dict" not in state
    assert state["feature_names"] == ["goal", "terminal_success"]
    torch.testing.assert_close(state["feature_scales"], torch.tensor([2.0, 1.0]))
    restored = _learner()
    restored.load_state(state)
    torch.testing.assert_close(
        restored.reward_model.reward.weight, learner.reward_model.reward.weight
    )


def test_exact_checkpoint_rejects_mismatched_basis_and_accepts_legacy_metadata() -> (
    None
):
    learner = _learner()
    state = learner.save_state()
    with pytest.raises(ValueError, match="feature names"):
        learner.load_state({**state, "feature_names": ["wrong", "terminal_success"]})
    with pytest.raises(ValueError, match="feature scales"):
        learner.load_state({**state, "feature_scales": torch.tensor([5.0, 1.0])})
    legacy_state = {
        name: value
        for name, value in state.items()
        if name not in {"feature_names", "feature_scales"}
    }
    learner.load_state(legacy_state)
    if torch.cuda.is_available():
        learner.load_state(
            {**state, "feature_scales": state["feature_scales"].to("cuda")}
        )
