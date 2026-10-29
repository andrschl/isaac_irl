"""The calibrated spread must never rewrite the successful MSE mean policy."""

import math

import pytest
import torch

from algorithms.off_policy.actor import (
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
)
from storage.expert_dataset import ExpertObsActionDataset
from training.behavior_cloning import save_actor
from training.spread_calibration import (
    fit_spread,
    interior_actions,
    load_mse_mean,
    spread_metrics,
)


def _actor(kind: str) -> StateDependentTanhPolicy:
    return StateDependentTanhPolicy(
        StateDependentTanhGaussianActor(
            3,
            2,
            hidden_dims=(8,),
            action_bound=4,
            initial_std=0.1,
            minimum_log_std=math.log(0.005),
            maximum_log_std=math.log(2),
            std_parameterization=kind,
        )
    )


def test_only_executable_interior_actions_have_a_tanh_density():
    actions = torch.tensor([[4.0, 0.0], [3.999, -1.0], [-4.0, 1.0]])
    assert interior_actions(actions, bound=4).tolist() == [False, True, False]
    with pytest.raises(ValueError, match="exceeds"):
        interior_actions(torch.tensor([[4.1, 0.0]]), bound=4)


def test_spread_fit_preserves_mse_mean_and_ignores_clipped_actions(tmp_path):
    torch.manual_seed(3)
    original = _actor("clamped_log")
    checkpoint = tmp_path / "mean.pt"
    save_actor(
        original, str(checkpoint), facts={"loss_type": "mse", "gradient_steps": 100}
    )
    calibrated = _actor("smooth_bounded")
    load_mse_mean(calibrated, checkpoint, device="cpu")
    observations = torch.randn(32, 3)
    expert_actions = original.deterministic_action(observations).detach()
    expert_actions[0, 1] = 4.0
    dataset = ExpertObsActionDataset([({"policy": observations}, expert_actions)])
    mean_before = (
        calibrated.deterministic_action({"policy": observations}).detach().clone()
    )
    std_before = calibrated.model.std_head.bias.detach().clone()
    fit_spread(
        calibrated,
        dataset,
        gradient_steps=20,
        batch_size=16,
        learning_rate=1e-3,
        final_learning_rate=1e-5,
    )
    torch.testing.assert_close(
        calibrated.deterministic_action({"policy": observations}),
        mean_before,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(calibrated.model.std_head.bias, std_before)
    metrics = spread_metrics(calibrated, dataset, device="cpu")
    assert metrics["excluded_boundary_pairs"] == 1
    assert metrics["interior_pairs"] == 31
    assert metrics["action_mse"] >= 0
