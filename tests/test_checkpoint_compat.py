"""Starting a run from a checkpoint this repository wrote."""

from __future__ import annotations

import pytest


def test_actor_initialization_reads_an_off_policy_checkpoint(tmp_path) -> None:
    """A run must be able to start from a policy this repository trained."""
    torch = pytest.importorskip("torch")
    from adapters.checkpoints import load_actor_only_checkpoint
    from algorithms.off_policy.actor import StateDependentTanhGaussianActor

    trained = StateDependentTanhGaussianActor(3, 2, hidden_dims=(8,))
    checkpoint = tmp_path / "model_10.pt"
    torch.save(
        {"rl_alg_state": {"learner": {"policy": trained.state_dict()}}}, checkpoint
    )
    fresh = StateDependentTanhGaussianActor(3, 2, hidden_dims=(8,))

    load_actor_only_checkpoint(fresh, str(checkpoint), device="cpu")

    torch.testing.assert_close(fresh.mean_head.weight, trained.mean_head.weight)


def test_actor_initialization_rejects_a_checkpoint_without_an_actor(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from adapters.checkpoints import load_actor_only_checkpoint
    from algorithms.off_policy.actor import StateDependentTanhGaussianActor

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"rl_alg_state": {"learner": {"value": {}}}}, checkpoint)

    with pytest.raises(ValueError, match="no actor in the checkpoint"):
        load_actor_only_checkpoint(
            StateDependentTanhGaussianActor(3, 2, hidden_dims=(8,)),
            str(checkpoint),
            device="cpu",
        )
