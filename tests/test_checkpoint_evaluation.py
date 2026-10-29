"""Evaluation must measure complete episodes without becoming another learner."""

from types import SimpleNamespace

import pytest
import torch
from conftest import build_rollout

from algorithms.off_policy.values.diagnostics import (
    successor_value_diagnostics,
    undiscounted_returns,
)
from config.cli import build_checkpoint_evaluation_parser
from config.schema import RewardModelCfg
from reward.model import LinearFeatureRewardModel
from storage.episode_batch import EpisodeBatch
from training.checkpoint_evaluation import (
    evaluate_checkpoint,
    evaluation_metrics,
    load_bc_members,
    load_evaluation_checkpoint,
)


def _episodes():
    rollout = build_rollout(num_envs=2, horizon=3, feature_dim=2)
    rollout.features = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 50.0]],
            [[7.0, 0.0], [999.0, 999.0], [999.0, 999.0]],
        ]
    )
    rollout.terminated.zero_()
    rollout.truncated.zero_()
    rollout.terminated[1, 0] = True
    rollout.truncated[0, -1] = True
    rollout.bootstrap_valid = ~(rollout.terminated | rollout.truncated)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    return EpisodeBatch(rollout, mask, 6)


def test_suffix_returns_include_terminal_reward_and_exclude_padding():
    batch = _episodes()
    returns = undiscounted_returns(batch.transitions.features, batch.mask)
    torch.testing.assert_close(
        returns,
        torch.tensor(
            [
                [[6.0, 50.0], [5.0, 50.0], [3.0, 50.0]],
                [[7.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        ),
    )


def test_raw_value_diagnostics_entropy_excludes_current_step_and_terminal_bootstrap():
    batch = _episodes()
    returns = undiscounted_returns(batch.transitions.features, batch.mask)[batch.mask]
    predictions = returns[None].repeat(3, 1, 1)
    value = SimpleNamespace(
        train_cfg=SimpleNamespace(gamma=1.0),
        control_quantile=0.5,
        online=lambda obs, actions: predictions,
        reward=SimpleNamespace(
            step_dt=torch.tensor(0.02), weights=torch.tensor([-2.0, 1.0])
        ),
        entropy=SimpleNamespace(
            value=lambda obs, actions: torch.tensor([0.04, 0.02, 0.0, 0.0])
        ),
    )
    policy = SimpleNamespace(log_probability=lambda obs, actions: -torch.ones(4))
    metrics = successor_value_diagnostics(
        value,
        policy,
        batch,
        entropy_rate=0.006,
        feature_names=("dense", "terminal_success"),
    )
    for name, result in metrics.items():
        if name.endswith(("bias", "rmse", "head_std_mean")):
            assert result == pytest.approx(0.0, abs=1e-7), name
    assert metrics["sparse/start/predicted_mean"] == pytest.approx(0.5)
    assert metrics["reward/start/predicted_mean"] == pytest.approx(0.24)
    value.train_cfg.gamma = 0.99
    with pytest.raises(ValueError, match="gamma=1"):
        successor_value_diagnostics(
            value,
            policy,
            batch,
            entropy_rate=0.0,
            feature_names=("dense", "terminal_success"),
        )


@pytest.mark.parametrize("diagnose_values", [True, False])
@pytest.mark.parametrize("modes", [("deterministic", "stochastic"), ("deterministic",)])
def test_checkpoint_evaluation_uses_collection_only_and_diagnostics_only_on_stochastic(
    monkeypatch,
    diagnose_values,
    modes,
):
    batch = _episodes()
    calls = []
    actor = SimpleNamespace(
        act_inference=lambda obs: torch.zeros(2, 1),
        sample_with_log_probability=lambda obs, generator: (
            torch.zeros(2, 1),
            torch.zeros(2),
        ),
    )
    value = SimpleNamespace()
    monkeypatch.setattr(
        "training.checkpoint_evaluation.SuccessorActionValue", SimpleNamespace
    )

    def collect(**kwargs):
        calls.append(("collect", kwargs["policy_version"], kwargs["episodes"]))
        kwargs["action_fn"](torch.zeros(2, 1))
        return batch

    def diagnostics(*args, **kwargs):
        calls.append(("diagnostics",))
        return {"reward/start/bias": 0.0}

    monkeypatch.setattr(
        "training.checkpoint_evaluation.successor_value_diagnostics", diagnostics
    )
    algorithm = SimpleNamespace(
        actor=actor,
        learner=SimpleNamespace(value=value, policy_version=7, entropy_rate=0.006),
        replay=SimpleNamespace(transition_count=0, demonstration_count=0),
        eval_mode=lambda: None,
        collector=SimpleNamespace(
            collect=collect,
            steps=SimpleNamespace(
                device=torch.device("cpu"),
                feature_tap=SimpleNamespace(
                    spec=SimpleNamespace(names=("dense", "terminal_success"))
                ),
            ),
        ),
    )
    rows = evaluate_checkpoint(
        algorithm,
        env=SimpleNamespace(seed=lambda seed: calls.append(("seed", seed))),
        episodes=2,
        seeds=[1001, 1002],
        diagnose_values=diagnose_values,
        modes=modes,
    )
    assert len(rows) == 2 * len(modes)
    assert len([call for call in calls if call[0] == "collect"]) == len(rows)
    assert len([call for call in calls if call[0] == "diagnostics"]) == (
        2 if diagnose_values and "stochastic" in modes else 0
    )
    assert all(row["successes"] == 1 for row in rows)
    assert sum(row["environment_transitions"] for row in rows) == 6 * len(rows)
    assert sum(row["discarded_transitions"] for row in rows) == 2 * len(rows)
    result = {
        "training_environment_transitions": 80000,
        "training_episodes": 320,
        "rows": rows,
    }
    metrics = evaluation_metrics(
        result, cumulative_evaluation_interactions=6 * len(rows)
    )
    assert metrics["Samples/total_environment_interactions"] == 80000 + 6 * len(rows)
    assert metrics["Evaluation/deterministic/success_rate"] == 0.5
    assert ("Evaluation/stochastic/success_rate" in metrics) == ("stochastic" in modes)
    assert any(key.startswith("Values/") for key in metrics) == (
        diagnose_values and "stochastic" in modes
    )


def test_actor_only_checkpoint_has_no_trained_values_or_online_interactions():
    from algorithms.off_policy.actor import (
        StateDependentTanhGaussianActor,
        StateDependentTanhPolicy,
    )

    actor = StateDependentTanhPolicy(
        StateDependentTanhGaussianActor(3, 2, hidden_dims=(4,))
    )
    source = StateDependentTanhPolicy(
        StateDependentTanhGaussianActor(3, 2, hidden_dims=(4,))
    )
    algorithm = SimpleNamespace(actor=actor, learner=SimpleNamespace(policy_version=10))
    checkpoint = {
        "actor_state_dict": source.model.state_dict(),
        "behavior_cloning": {"gradient_steps": 500},
    }
    metadata = load_evaluation_checkpoint(algorithm, None, checkpoint)
    assert not metadata["has_value_estimates"]
    assert (
        metadata["training_environment_transitions"]
        == metadata["training_episodes"]
        == 0
    )
    for key, expected in source.model.state_dict().items():
        torch.testing.assert_close(actor.model.state_dict()[key], expected)
        torch.testing.assert_close(
            algorithm.learner.previous_policy.model.state_dict()[key], expected
        )


def test_bc_member_loading_requires_shared_demonstration_and_policy_contract(
    tmp_path,
):
    from algorithms.off_policy.actor import (
        StateDependentTanhGaussianActor,
        StateDependentTanhPolicy,
    )

    template = StateDependentTanhPolicy(
        StateDependentTanhGaussianActor(3, 2, hidden_dims=(4,))
    )
    first_path = tmp_path / "member_1.pt"
    second_path = tmp_path / "member_2.pt"
    facts = {
        "policy": {"action_bound": 1.0},
        "expert_data_path": str(tmp_path / "demos.hdf5"),
        "expert_pairs": 20,
    }
    for path in (first_path, second_path):
        torch.save(
            {
                "actor_state_dict": template.model.state_dict(),
                "behavior_cloning": facts,
            },
            path,
        )
    members, member_facts = load_bc_members(
        template, [first_path, second_path], device="cpu"
    )
    assert len(members) == len(member_facts) == 2
    with pytest.raises(ValueError, match="distinct checkpoints"):
        load_bc_members(template, [first_path, first_path], device="cpu")
    different_facts = {**facts, "expert_data_path": str(tmp_path / "other.hdf5")}
    torch.save(
        {
            "actor_state_dict": template.model.state_dict(),
            "behavior_cloning": different_facts,
        },
        second_path,
    )
    with pytest.raises(ValueError, match="share policy, demonstrations"):
        load_bc_members(template, [first_path, second_path], device="cpu")


def test_evaluation_rejects_reward_basis_mismatch_before_loading_values():
    calls = []
    algorithm = SimpleNamespace(
        learner=SimpleNamespace(load_state=lambda *args, **kwargs: calls.append("load"))
    )
    reward_model = LinearFeatureRewardModel(
        RewardModelCfg(is_linear=True), num_features=2, feature_scales=(5.0, 1.0)
    )
    checkpoint = {
        "reward_learner_state": {
            "feature_names": ["goal", "terminal_success"],
            "feature_scales": torch.tensor([5.0, 1.0]),
        }
    }
    with pytest.raises(ValueError, match="learned-feature names"):
        load_evaluation_checkpoint(
            algorithm,
            reward_model,
            checkpoint,
            reward_feature_names=("wrong", "terminal_success"),
        )
    assert calls == []
    checkpoint["reward_learner_state"]["feature_scales"] = torch.tensor([2.0, 1.0])
    with pytest.raises(ValueError, match="feature scales"):
        load_evaluation_checkpoint(
            algorithm,
            reward_model,
            checkpoint,
            reward_feature_names=("goal", "terminal_success"),
        )
    assert calls == []
    if torch.cuda.is_available():
        checkpoint["reward_learner_state"].update(
            feature_scales=torch.tensor([5.0, 1.0], device="cuda"),
            reward_model_state_dict=reward_model.state_dict(),
        )
        checkpoint["rl_alg_state"] = {"learner": {}}
        checkpoint["interaction_counts"] = {
            "environment_transitions": 0,
            "evaluation_transitions": 0,
            "imitator_transitions": 0,
            "training_episodes": 0,
        }
        checkpoint["global_timestep"] = 0
        load_evaluation_checkpoint(
            algorithm,
            reward_model,
            checkpoint,
            reward_feature_names=("goal", "terminal_success"),
        )
        assert calls == ["load"]


def test_checkpoint_evaluation_cli_defaults_to_independent_reset_seeds():
    args = build_checkpoint_evaluation_parser().parse_args(
        [
            "--task",
            "lift",
            "--config",
            "experiment.yaml",
            "--checkpoints",
            "model_0.pt",
            "model_10.pt",
            "--output_dir",
            "output",
        ]
    )
    assert args.evaluation_seeds == [1001, 1002]
    assert args.episodes_per_seed == 32
    assert args.modes == ["deterministic", "stochastic"]
    assert args.checkpoints == ["model_0.pt", "model_10.pt"]
