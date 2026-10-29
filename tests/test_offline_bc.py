"""Offline behavioral cloning, and the actor every online arm starts from."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")

from conftest import ACTION_DIM, OBS_DIM, FakeIsaacEnv

from adapters.off_policy_factory import build_off_policy_stack
from algorithms.bc import BC
from algorithms.off_policy.actor import (
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
)
from config.schema import (
    BCCfg,
    DirectQCfg,
    OffPolicyTrainCfg,
    PmdCfg,
    TanhGaussianPolicyCfg,
    TransitionReplayCfg,
)
from storage.expert_dataset import ExpertObsActionDataset
from training.bc_regularization import episode_folds
from training.bc_regularization import run as validate_bc
from training.behavior_cloning import actor_fit_metrics, fit_actor, mean_policy_l2, run

HORIZON = 6
EPISODES = 3


def _write_demos(path: Path) -> str:
    """A demonstration file with the (obs, action) pairs BC reads."""
    h5py = pytest.importorskip("h5py")
    generator = torch.Generator().manual_seed(0)
    with h5py.File(path, "w", track_order=True) as handle:
        data = handle.create_group("data", track_order=True)
        for episode in range(EPISODES):
            demo = data.create_group(f"demo_{episode}", track_order=True)
            demo.create_group("obs").create_dataset(
                "policy",
                data=torch.randn(HORIZON, OBS_DIM, generator=generator).numpy(),
            )
            demo.create_dataset(
                "actions",
                data=torch.randn(HORIZON, ACTION_DIM, generator=generator)
                .tanh()
                .numpy(),
            )
    return str(path)


def _write_config(root: Path, demos: str) -> str:
    payload = {
        "seed": 7,
        "device": "cpu",
        "env": {"name": "Isaac-Lift-Cube-Franka-v0", "num_envs": 4},
        "expert_data": {"path": demos},
        "reward": {"source": "environment"},
        "algo": {
            "type": "pmd",
            "pmd": {
                "policy": {
                    "hidden_dims": [8],
                    "initial_std": 0.1,
                    "minimum_std": 0.05,
                    "action_bound": 4.0,
                },
                "train": {
                    "episodes_per_batch": 4,
                    "critic_batch_size": 8,
                    "critic_updates_per_batch": 4,
                    "initial_critic_updates": 0,
                    "replay": {"learning_starts": 1, "capacity": 1024},
                },
                "value": {"type": "direct", "num_heads": 2, "width": 16},
            },
        },
    }
    path = root / "bc.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(path)


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    demos = _write_demos(tmp_path / "demos.hdf5")
    settings = dict(
        task="Isaac-Lift-Cube-Franka-v0",
        config=_write_config(tmp_path, demos),
        set=[],
        seed=None,
        num_envs=None,
        device=None,
        expert_data_path=None,
        expert_num_trajectories=None,
        gradient_steps=120,
        learning_rate=3.0e-4,
        final_learning_rate=None,
        l2_weight=0.0,
        loss_type="mse",
        validation_fraction=0.0,
        batch_size=16,
        log_interval=0,
        output=str(tmp_path / "bc_actor.pt"),
    )
    settings.update(overrides)
    return argparse.Namespace(**settings)


def _stack_cfg() -> PmdCfg:
    return PmdCfg(
        train=OffPolicyTrainCfg(
            episodes_per_batch=4,
            critic_batch_size=8,
            critic_updates_per_batch=4,
            initial_critic_updates=0,
            replay=TransitionReplayCfg(learning_starts=1, capacity=1024),
        ),
        policy=TanhGaussianPolicyCfg(
            hidden_dims=(8,), initial_std=0.1, minimum_std=0.05, action_bound=4.0
        ),
        value=DirectQCfg(num_heads=2, width=16, residual_blocks=1),
        actor_batch_size=8,
        candidate_actions=4,
        projection_steps=2,
        policy_improvements_per_batch=1,
    )


def test_offline_cloning_fits_the_actor_without_touching_a_simulator(tmp_path) -> None:
    """Cloning happens once, before any run, and costs no environment steps."""
    args = _args(tmp_path)

    result = run(args)

    assert result.expert_pairs == EPISODES * HORIZON
    assert result.gradient_steps == 120
    assert Path(result.checkpoint_path).exists()
    # The bounded actor is the online one: same bound, same shapes.
    assert float(result.policy.model.action_bound) == 4.0
    assert result.policy.model.observation_dim == OBS_DIM


def test_cloning_actually_moves_the_actor_towards_the_demonstrations(tmp_path) -> None:
    short = run(_args(tmp_path / "short", gradient_steps=5))
    long = run(_args(tmp_path / "long", gradient_steps=400))

    assert long.final_loss < short.final_loss


def test_refinement_loads_the_actor_before_any_bc_step(tmp_path, monkeypatch) -> None:
    original = run(_args(tmp_path / "original"))
    expected = original.policy.model.state_dict()

    def check_initialization(policy, behavior_cloning, **kwargs):
        assert kwargs["learning_rate"] == 3.0e-5
        for name, value in policy.model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        return 0.0

    monkeypatch.setattr("training.behavior_cloning.fit_actor", check_initialization)
    refined = run(
        _args(
            tmp_path / "refined",
            init_policy_checkpoint=original.checkpoint_path,
            learning_rate=3.0e-5,
        )
    )
    saved = torch.load(refined.checkpoint_path, weights_only=False)
    assert (
        saved["behavior_cloning"]["init_policy_checkpoint"] == original.checkpoint_path
    )


@pytest.mark.parametrize("std_parameterization", ["clamped_log", "smooth_bounded"])
def test_every_online_arm_starts_from_exactly_the_cloned_actor(
    tmp_path, std_parameterization
) -> None:
    """The comparison is between reward signals, not between starting points.

    PMD's first improvement is a projection away from its *previous* policy, so
    a warm start that reached the actor but not that snapshot would let the
    first update pull the policy back towards a random initialization.
    """
    result = run(
        _args(
            tmp_path,
            set=[f"algo.pmd.policy.std_parameterization={std_parameterization}"],
        )
    )
    cfg = _stack_cfg()
    cfg = replace(
        cfg, policy=replace(cfg.policy, std_parameterization=std_parameterization)
    )

    stack = build_off_policy_stack(
        env=FakeIsaacEnv(),
        observations=FakeIsaacEnv().get_observations(),
        algorithm_cfg=cfg,
        ignored_reward_terms=set(),
        device="cpu",
        seed=0,
        init_policy_checkpoint=result.checkpoint_path,
    )

    cloned = result.policy.model.state_dict()
    for name, value in stack.algorithm.actor.model.state_dict().items():
        torch.testing.assert_close(value, cloned[name], rtol=0, atol=0)
    for (
        name,
        value,
    ) in stack.algorithm.learner.previous_policy.model.state_dict().items():
        torch.testing.assert_close(value, cloned[name], rtol=0, atol=0)


def test_an_actor_that_cannot_be_the_online_policy_is_refused(tmp_path) -> None:
    args = _args(tmp_path)
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    payload["algo"]["type"] = "ppo"
    Path(args.config).write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="algo.type=pmd"):
        run(args)


def _tiny_fit():
    torch.manual_seed(12)
    policy = StateDependentTanhPolicy(
        StateDependentTanhGaussianActor(
            3,
            2,
            hidden_dims=(8,),
            initial_std=0.1,
        )
    )
    storage = ExpertObsActionDataset(
        [({"policy": torch.randn(16, 3)}, torch.zeros(16, 2))]
    )
    bc = BC(
        cfg=BCCfg(weight=1.0, loss_type="mse", batch_size=8, val_fraction=0),
        storage=storage,
    )
    return policy, bc


def test_bc_l2_penalizes_only_mean_matrices() -> None:
    policy, _ = _tiny_fit()
    penalty = mean_policy_l2(policy)
    expected = 0.5 * sum(
        parameter.square().sum()
        for name, parameter in policy.model.named_parameters()
        if name.endswith("weight") and not name.startswith("log_std_head")
    )
    torch.testing.assert_close(penalty, expected)
    penalty.backward()
    for name, parameter in policy.model.named_parameters():
        if name.endswith("weight") and not name.startswith("log_std_head"):
            torch.testing.assert_close(parameter.grad, parameter)
        else:
            assert parameter.grad is None


def test_regularized_fit_spends_full_budget_and_preserves_exploration() -> None:
    policy, bc = _tiny_fit()
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.model.log_std_head.named_parameters()
    }
    trace = []
    fit_actor(
        policy,
        bc,
        gradient_steps=7,
        learning_rate=3e-4,
        final_learning_rate=3e-6,
        l2_weight=1e-3,
        log_interval=1,
        on_metrics=lambda step, metrics: trace.append((step, metrics)),
    )
    assert [step for step, _ in trace] == list(range(8))
    assert trace[-1][1]["learning_rate"] == pytest.approx(3e-6)
    for name, parameter in policy.model.log_std_head.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    metrics = trace[-1][1]
    assert metrics["training_objective"] == pytest.approx(
        metrics["training_mse"] + metrics["regularization"]
    )


def test_validation_metrics_do_not_change_fit_or_rng() -> None:
    policy, bc = _tiny_fit()
    fit_actor(policy, bc, gradient_steps=5, learning_rate=3e-4)
    expected = {
        name: value.clone() for name, value in policy.model.state_dict().items()
    }
    policy, bc = _tiny_fit()
    bc.val_storage = ExpertObsActionDataset(
        [({"policy": torch.ones(4, 3)}, torch.full((4, 2), 100.0))]
    )
    state = torch.random.get_rng_state()
    metrics = actor_fit_metrics(policy, bc, l2_weight=0)
    assert metrics["validation_mse"] > 9000
    assert torch.equal(state, torch.random.get_rng_state())
    fit_actor(policy, bc, gradient_steps=5, learning_rate=3e-4, log_interval=1)
    for name, value in policy.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"l2_weight": -1},
        {"l2_weight": float("nan")},
        {"final_learning_rate": 1.0},
        {"learning_rate": 0.0},
    ],
)
def test_invalid_bc_regularization_settings_are_rejected(kwargs) -> None:
    policy, bc = _tiny_fit()
    options = {"gradient_steps": 2, "learning_rate": 3e-4, **kwargs}
    with pytest.raises(ValueError):
        fit_actor(policy, bc, **options)


def test_episode_folds_are_disjoint_complete_and_reproducible() -> None:
    folds = episode_folds(10, 5, seed=42)
    assert folds == episode_folds(10, 5, seed=42)
    assert sorted(index for _, held_out in folds for index in held_out) == list(
        range(10)
    )
    for training, validation in folds:
        assert len(training) == 8 and len(validation) == 2
        assert not set(training).intersection(validation)
        assert set(training + validation) == set(range(10))


def test_cross_validation_selects_final_loss_and_refits_all_demos(tmp_path) -> None:
    args = _args(
        tmp_path,
        gradient_steps=2,
        folds=3,
        l2_candidates=[0.0, 1e-4],
        set=["logger=noop"],
    )
    summary = validate_bc(args)
    checkpoint = torch.load(args.output, weights_only=False)
    facts = checkpoint["behavior_cloning"]
    assert facts["training_episodes"] == list(range(EPISODES))
    assert facts["validation_episodes"] == []
    assert facts["expert_pairs"] == EPISODES * HORIZON
    assert facts["l2_weight"] == min(
        summary["mean_validation_mse"], key=summary["mean_validation_mse"].get
    )
    assert len(summary["fold_scores"]) == 6
    assert all(row["fit_trace"][-1]["step"] == 2 for row in summary["fold_scores"])


def test_likelihood_validation_uses_nll_and_refits_all_episodes(tmp_path) -> None:
    args = _args(
        tmp_path,
        loss_type="nll",
        gradient_steps=3,
        folds=3,
        l2_candidates=[0.0, 1e-4],
        set=[
            "logger=noop",
            "algo.pmd.policy.std_parameterization=smooth_bounded",
            "algo.pmd.policy.minimum_std=0.005",
        ],
    )
    summary = validate_bc(args)
    facts = torch.load(args.output, weights_only=False)["behavior_cloning"]
    assert "mean_validation_nll" in summary
    assert facts["loss_type"] == "nll"
    assert facts["training_episodes"] == list(range(EPISODES))
    assert facts["validation_episodes"] == []
    assert facts["policy"]["std_parameterization"] == "smooth_bounded"
    assert facts["training_objective"] == pytest.approx(
        facts["training_nll"] + facts["regularization"]
    )
    assert facts["training_std_min"] >= 0.005


def test_offline_nll_updates_spread_and_reports_the_correct_objective(tmp_path):
    args = _args(
        tmp_path,
        loss_type="nll",
        gradient_steps=10,
        set=[
            "algo.pmd.policy.std_parameterization=smooth_bounded",
            "algo.pmd.policy.minimum_std=0.005",
        ],
    )
    result = run(args)
    facts = torch.load(args.output, weights_only=False)["behavior_cloning"]
    assert not torch.allclose(
        result.policy.model.std_head.weight,
        torch.zeros_like(result.policy.model.std_head.weight),
    )
    assert facts["training_objective"] == pytest.approx(facts["training_nll"])
    assert facts["training_mse"] >= 0
