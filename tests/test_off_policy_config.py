from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
import yaml

from config import DirectQCfg, PmdCfg, SacCfg, SuccessorQCfg, load_train_cfg
from config.benchmark_cli import build_benchmark_cfg


def _train_args(**overrides):
    values = {"task": "Isaac-Test-v0"}
    values.update(overrides)
    return argparse.Namespace(**values)


def test_loads_sac_with_direct_q_and_bc_weight(tmp_path) -> None:
    payload = {
        "env": {"name": "Isaac-Test-v0", "device": "cpu", "num_envs": 16},
        "algo": {
            "type": "sac",
            "actor_learning_rate": 1.0e-3,
            "train": {
                "gamma": 0.97,
                "updates_per_transition": 0.25,
                "replay": {"learning_starts": 128, "capacity": 1024},
            },
            "value": {"type": "direct", "num_heads": 3},
        },
        "bc": {"weight": 0.1},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(payload))

    cfg = load_train_cfg(
        task_name="Isaac-Test-v0",
        args_cli=_train_args(),
        config_dir=tmp_path,
    )

    assert isinstance(cfg.rl_algorithm, SacCfg)
    assert isinstance(cfg.rl_algorithm.value, DirectQCfg)
    assert cfg.rl_algorithm.value.num_heads == 3
    assert cfg.rl_algorithm.train.gamma == pytest.approx(0.97)
    assert cfg.rl_algorithm.train.updates_per_transition == pytest.approx(0.25)
    assert cfg.bc.weight == pytest.approx(0.1)


def test_loads_pmd_with_successor_q_and_projection_parameters(tmp_path) -> None:
    payload = {
        "env": {"name": "Isaac-Test-v0", "device": "cpu"},
        "algo": {
            "type": "pmd",
            "beta": 0.4,
            "candidate_target_kl": 0.03,
            "actor_projection_kl": 0.01,
            "critic_updates_per_projection": 4,
            "value": {"type": "successor", "num_heads": 5},
        },
        "bc": {"weight": 0.0},
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(payload))

    cfg = load_train_cfg(
        task_name="Isaac-Test-v0",
        args_cli=_train_args(),
        config_dir=tmp_path,
    )

    assert isinstance(cfg.rl_algorithm, PmdCfg)
    assert isinstance(cfg.rl_algorithm.value, SuccessorQCfg)
    assert cfg.rl_algorithm.beta == pytest.approx(0.4)
    assert cfg.rl_algorithm.candidate_target_kl == pytest.approx(0.03)
    assert cfg.rl_algorithm.actor_projection_kl == pytest.approx(0.01)
    assert cfg.rl_algorithm.critic_updates_per_projection == 4


def test_algorithm_cli_override_keeps_nested_yaml_values(tmp_path) -> None:
    payload = {
        "env": {"name": "Isaac-Test-v0", "device": "cpu"},
        "algo": {
            "type": "sac",
            "actor_learning_rate": 7.0e-4,
            "train": {"gamma": 0.96},
            "value": {"type": "direct", "num_heads": 2},
        },
    }
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(payload))

    cfg = load_train_cfg(
        task_name="Isaac-Test-v0",
        args_cli=_train_args(algorithm="pmd"),
        config_dir=tmp_path,
    )

    assert isinstance(cfg.rl_algorithm, PmdCfg)
    assert cfg.rl_algorithm.actor_learning_rate == pytest.approx(7.0e-4)
    assert cfg.rl_algorithm.train.gamma == pytest.approx(0.96)
    assert isinstance(cfg.rl_algorithm.value, DirectQCfg)
    assert cfg.rl_algorithm.value.num_heads == 2


def test_removed_algorithm_names_are_rejected(tmp_path) -> None:
    (tmp_path / "experiment.yaml").write_text(
        yaml.safe_dump(
            {
                "env": {"device": "cpu"},
                "algo": {"type": "off_policy_grpo"},
            }
        )
    )

    with pytest.raises(ValueError, match="ppo, sac, pmd"):
        load_train_cfg(
            task_name="Isaac-Test-v0",
            args_cli=_train_args(),
            config_dir=tmp_path,
        )


def test_benchmark_cli_overrides_only_the_named_axes() -> None:
    args = SimpleNamespace(
        algorithm="pmd",
        task="cartpole",
        seed=1,
        num_envs=16,
        transition_budget=10_000,
        value_backend="direct",
        num_heads=2,
        updates_per_transition=0.25,
        pmd_beta=0.0,
        pmd_critic_updates_per_projection=4,
    )

    cfg = build_benchmark_cfg(args)

    assert cfg.algorithm == "pmd"
    assert cfg.task.name == "cartpole"
    assert cfg.task.num_envs == 16
    assert cfg.task.transition_budget == 10_000
    assert isinstance(cfg.task.pmd.value, DirectQCfg)
    assert cfg.task.pmd.value.num_heads == 2
    # Selecting a backend must not silently drop the profile's architecture.
    assert cfg.task.pmd.value.width == 256
    assert cfg.task.pmd.value.residual_blocks == 1
    assert cfg.task.pmd.beta == 0.0
    assert cfg.task.pmd.critic_updates_per_projection == 4
    assert cfg.task.pmd.train.updates_per_transition == pytest.approx(0.25)
    # Untouched axes keep the task profile.
    assert cfg.task.evaluation_steps == 250
    assert cfg.task.pmd.policy.action_bound == pytest.approx(1.0)


def test_lift_profile_carries_its_own_baseline_and_success_metric() -> None:
    args = SimpleNamespace(algorithm="ppo", task="lift")

    cfg = build_benchmark_cfg(args)

    assert cfg.task.task_id == "Isaac-Lift-Cube-Franka-v0"
    assert cfg.task.ppo.rollout_steps == 24
    assert cfg.task.ppo.hidden_dims == (256, 128, 64)
    assert cfg.task.success_threshold == pytest.approx(0.08)
    assert cfg.task.success_rate_thresholds == (0.25, 0.5, 0.9)
    assert "success_bonus" in cfg.task.force_include_reward_terms
    assert cfg.group == "lift-fixed-reward-v4"


def test_unknown_benchmark_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark task"):
        build_benchmark_cfg(SimpleNamespace(algorithm="ppo", task="pendulum"))
