from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from algorithms import IRLCfg
from config import TrainCfg, load_train_cfg
from reward_model import RewardModelCfg
from runner import IrlRunnerCfg


def _load_train_irl_module():
    """Compatibility shim — only used for tests that touch script-side helpers
    (e.g. ``_dump_run_configs``) that still live in ``scripts/irl/train_irl.py``.
    Pure config tests import from ``src/config`` directly."""
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "irl" / "train_irl.py"
    module_name = "train_irl_module_config_parser"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides) -> argparse.Namespace:
    payload = {
        "task": "Isaac-Lift-Cube-Franka-v0",
        "seed": None,
        "max_iterations": None,
        "device": None,
        "num_envs": None,
        "resume": None,
        "load_run": None,
        "checkpoint": None,
        "run_name": None,
        "experiment_name": None,
        "logger": None,
        "log_project_name": None,
        "expert_data_path": None,
        "expert_num_trajectories": None,
        "expert_subset_strategy": None,
        "irl_discount_gamma": None,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_load_train_cfg_loads_from_experiment_yaml():
    """Load from configs/franka_lift/experiment.yaml (single experiment config)."""
    module = _load_train_irl_module()
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "franka_lift"
    cfg = load_train_cfg(
        task_name="Isaac-Lift-Cube-Franka-v0",
        args_cli=_args(),
        config_dir=config_dir,
    )

    assert cfg.experiment_name == "franka_lift"
    assert isinstance(cfg.runner, IrlRunnerCfg)
    assert isinstance(cfg.irl, IRLCfg)
    assert isinstance(cfg.reward, RewardModelCfg)
    assert cfg.runner.rl_updates_per_cycle == 1
    assert cfg.runner.reward_updates_per_cycle == 1
    assert cfg.runner.validation_interval == 50
    assert cfg.runner.validation_steps_per_env is None
    assert cfg.irl.discount_gamma is None
    assert cfg.irl.normalize_returns_by_episode_length is False


def test_load_train_cfg_loads_env_from_experiment_yaml(tmp_path: Path):
    """experiment.yaml contains env section; env fields are parsed."""
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    experiment_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"name": "Isaac-Lift-Cube-Franka-v0", "device": "cpu", "num_envs": 8},
        "reward": {"type": "dense"},
        "policy": {},
        "algo": {},
        "irl": {},
        "runner": {},
        "feature_map": {},
    }
    (config_dir / "experiment.yaml").write_text(
        yaml.safe_dump(experiment_payload, sort_keys=False), encoding="utf-8"
    )

    cfg = load_train_cfg(
        task_name="Isaac-Lift-Cube-Franka-v0",
        args_cli=_args(task="Isaac-Lift-Cube-Franka-v0"),
        config_dir=config_dir,
    )
    assert cfg.env.name == "Isaac-Lift-Cube-Franka-v0"
    assert cfg.env.device == "cpu"
    assert cfg.env.num_envs == 8


def test_load_train_cfg_rejects_stale_runner_key(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"device": "cpu"},
        "runner": {"reward_update_interval": 5},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="reward_update_interval"):
        load_train_cfg(
            task_name="Isaac-Unit-Test-v0",
            args_cli=_args(task="Isaac-Unit-Test-v0"),
            config_dir=config_dir,
        )


def test_load_train_cfg_rejects_stale_bc_eval_interval_key(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"device": "cpu"},
        "runner": {"bc_eval_interval": 5},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="bc_eval_interval"):
        load_train_cfg(
            task_name="Isaac-Unit-Test-v0",
            args_cli=_args(task="Isaac-Unit-Test-v0"),
            config_dir=config_dir,
        )


def test_load_train_cfg_parses_runner_validation_settings(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"device": "cpu"},
        "runner": {
            "validation_interval": 7,
            "validation_steps_per_env": 40,
        },
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    cfg = load_train_cfg(
        task_name="Isaac-Unit-Test-v0",
        args_cli=_args(task="Isaac-Unit-Test-v0"),
        config_dir=config_dir,
    )
    assert cfg.runner.validation_interval == 7
    assert cfg.runner.validation_steps_per_env == 40


def test_load_train_cfg_parses_bc_val_fraction(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"device": "cpu"},
        "bc": {"alpha": 0.5, "val_fraction": 0.2},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    cfg = load_train_cfg(
        task_name="Isaac-Unit-Test-v0",
        args_cli=_args(task="Isaac-Unit-Test-v0"),
        config_dir=config_dir,
    )
    assert cfg.bc.val_fraction == 0.2


def test_load_train_cfg_rejects_invalid_bc_val_fraction(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"device": "cpu"},
        "bc": {"alpha": 0.5, "val_fraction": 1.0},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="val_fraction"):
        load_train_cfg(
            task_name="Isaac-Unit-Test-v0",
            args_cli=_args(task="Isaac-Unit-Test-v0"),
            config_dir=config_dir,
        )


@pytest.mark.parametrize(
    ("runner_payload", "error_match"),
    [
        ({"validation_interval": -1}, "validation_interval"),
        ({"validation_steps_per_env": -1}, "validation_steps_per_env"),
    ],
)
def test_load_train_cfg_rejects_negative_runner_validation_settings(
    tmp_path: Path,
    runner_payload: dict,
    error_match: str,
):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"device": "cpu"},
        "runner": runner_payload,
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        load_train_cfg(
            task_name="Isaac-Unit-Test-v0",
            args_cli=_args(task="Isaac-Unit-Test-v0"),
            config_dir=config_dir,
        )


def test_load_train_cfg_rejects_stale_buffer_key(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"device": "cpu"},
        "runner": {
            "imitator_buffer": {"store_discounted_feature_returns": False},
        },
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="store_discounted_feature_returns"):
        load_train_cfg(
            task_name="Isaac-Unit-Test-v0",
            args_cli=_args(task="Isaac-Unit-Test-v0"),
            config_dir=config_dir,
        )


def test_load_train_cfg_rejects_stale_reward_alias_keys(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"device": "cpu"},
        "reward": {
            "type": "dense",
            "reward_hidden_dims": [128, 64],
            "reward_is_linear": False,
        },
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="reward_hidden_dims"):
        load_train_cfg(
            task_name="Isaac-Unit-Test-v0",
            args_cli=_args(task="Isaac-Unit-Test-v0"),
            config_dir=config_dir,
        )


def test_load_train_cfg_rejects_removed_reward_gradient_mode_key(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"device": "cpu"},
        "irl": {"reward_gradient_mode": "invalid"},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="removed"):
        load_train_cfg(
            task_name="Isaac-Unit-Test-v0",
            args_cli=_args(task="Isaac-Unit-Test-v0"),
            config_dir=config_dir,
        )


def test_load_train_cfg_rejects_invalid_irl_discount_gamma_value(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"device": "cpu"},
        "irl": {"discount_gamma": 1.5},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="discount_gamma"):
        load_train_cfg(
            task_name="Isaac-Unit-Test-v0",
            args_cli=_args(task="Isaac-Unit-Test-v0"),
            config_dir=config_dir,
        )


def test_dump_run_configs_allows_unpickleable_env_cfg(tmp_path: Path):
    from utils.run_output import dump_run_configs

    run_dir = tmp_path / "run"

    # Lambdas are intentionally unpickleable in this context.
    env_cfg = {"callable": lambda x: x}
    train_cfg = TrainCfg()

    dump_run_configs(str(run_dir), env_cfg=env_cfg, train_cfg=train_cfg)

    assert (run_dir / "params" / "experiment.yaml").exists()
    assert not (run_dir / "params" / "experiment.pkl").exists()


def test_load_train_cfg_rejects_non_positive_max_iterations():
    module = _load_train_irl_module()
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "franka_lift"

    with pytest.raises(ValueError, match="max_iterations"):
        load_train_cfg(
            task_name="Isaac-Lift-Cube-Franka-v0",
            args_cli=_args(max_iterations=0),
            config_dir=config_dir,
        )


def test_load_train_cfg_cli_overrides_irl_discount_gamma():
    module = _load_train_irl_module()
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "franka_lift"

    cfg = load_train_cfg(
        task_name="Isaac-Lift-Cube-Franka-v0",
        args_cli=_args(irl_discount_gamma=0.7),
        config_dir=config_dir,
    )
    assert cfg.irl.discount_gamma == pytest.approx(0.7)


def test_load_train_cfg_parses_reward_regularization_and_projection(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"name": "Isaac-Lift-Cube-Franka-v0", "device": "cpu"},
        "reward": {
            "type": "dense",
            "hidden_dims": [16],
            "regularization": "l2",
            "regularization_strength": 0.01,
            "linear_projection": "none",
        },
        "policy": {},
        "algo": {},
        "irl": {},
        "runner": {},
        "feature_map": {},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    cfg = load_train_cfg(
        task_name="Isaac-Lift-Cube-Franka-v0",
        args_cli=_args(task="Isaac-Lift-Cube-Franka-v0"),
        config_dir=config_dir,
    )
    assert cfg.reward.regularization == "l2"
    assert cfg.reward.regularization_strength == pytest.approx(0.01)
    assert cfg.reward.linear_projection == "none"


def test_load_train_cfg_rejects_linear_projection_when_not_linear(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"name": "Isaac-Lift-Cube-Franka-v0", "device": "cpu"},
        "reward": {
            "type": "dense",
            "is_linear": False,
            "linear_projection": "l2_ball",
        },
        "policy": {},
        "algo": {},
        "irl": {},
        "runner": {},
        "feature_map": {},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="linear_projection"):
        load_train_cfg(
            task_name="Isaac-Lift-Cube-Franka-v0",
            args_cli=_args(task="Isaac-Lift-Cube-Franka-v0"),
            config_dir=config_dir,
        )


def test_load_train_cfg_type_linear_implies_is_linear(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"name": "Isaac-Lift-Cube-Franka-v0", "device": "cpu"},
        "reward": {
            "type": "linear",
            "regularization": "none",
            "linear_projection": "l2_ball",
        },
        "policy": {},
        "algo": {},
        "irl": {},
        "runner": {},
        "feature_map": {},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    cfg = load_train_cfg(
        task_name="Isaac-Lift-Cube-Franka-v0",
        args_cli=_args(task="Isaac-Lift-Cube-Franka-v0"),
        config_dir=config_dir,
    )
    assert cfg.reward.is_linear is True
    assert cfg.reward.regularization == "none"


def test_load_train_cfg_parses_expert_num_trajectories_and_subset_strategy(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"name": "Isaac-Lift-Cube-Franka-v0", "device": "cpu"},
        "reward": {"type": "dense"},
        "policy": {},
        "algo": {},
        "irl": {
            "expert_num_trajectories": 50,
            "expert_subset_strategy": "random",
        },
        "runner": {},
        "feature_map": {},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    cfg = load_train_cfg(
        task_name="Isaac-Lift-Cube-Franka-v0",
        args_cli=_args(task="Isaac-Lift-Cube-Franka-v0"),
        config_dir=config_dir,
    )
    assert cfg.irl.expert_num_trajectories == 50
    assert cfg.irl.expert_subset_strategy == "random"


def test_load_train_cfg_rejects_non_positive_expert_num_trajectories(tmp_path: Path):
    module = _load_train_irl_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    train_payload = {
        "experiment_name": "tmp",
        "seed": 1,
        "max_iterations": 10,
        "env": {"name": "Isaac-Lift-Cube-Franka-v0", "device": "cpu"},
        "reward": {"type": "dense"},
        "policy": {},
        "algo": {},
        "irl": {"expert_num_trajectories": 0},
        "runner": {},
        "feature_map": {},
    }
    (config_dir / "experiment.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="expert_num_trajectories"):
        load_train_cfg(
            task_name="Isaac-Lift-Cube-Franka-v0",
            args_cli=_args(task="Isaac-Lift-Cube-Franka-v0"),
            config_dir=config_dir,
        )
