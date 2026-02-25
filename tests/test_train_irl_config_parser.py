from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")


def _load_train_irl_module():
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
        "irl_discount_gamma": None,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_load_train_cfg_merges_base_and_task_override():
    module = _load_train_irl_module()
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    cfg = module.load_train_cfg(
        task_name="Isaac-Lift-Cube-Franka-v0",
        args_cli=_args(),
        config_dir=config_dir,
    )

    assert cfg.experiment_name == "franka_lift"
    assert isinstance(cfg.runner, module.IrlRunnerCfg)
    assert isinstance(cfg.irl, module.IRLCfg)
    assert isinstance(cfg.reward, module.RewardModelCfg)
    assert cfg.runner.policy_updates_per_cycle == 1
    assert cfg.runner.reward_updates_per_cycle == 1
    assert cfg.irl.discount_gamma is None
    assert cfg.irl.normalize_returns_by_episode_length is True


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
    (config_dir / "train.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="reward_update_interval"):
        module.load_train_cfg(
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
    (config_dir / "train.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="store_discounted_feature_returns"):
        module.load_train_cfg(
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
            "type": "dense_mlp",
            "reward_hidden_dims": [128, 64],
            "reward_is_linear": False,
        },
    }
    (config_dir / "train.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="reward_hidden_dims"):
        module.load_train_cfg(
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
    (config_dir / "train.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="removed"):
        module.load_train_cfg(
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
    (config_dir / "train.yaml").write_text(yaml.safe_dump(train_payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="discount_gamma"):
        module.load_train_cfg(
            task_name="Isaac-Unit-Test-v0",
            args_cli=_args(task="Isaac-Unit-Test-v0"),
            config_dir=config_dir,
        )


def test_dump_run_configs_allows_unpickleable_env_cfg(tmp_path: Path):
    module = _load_train_irl_module()
    run_dir = tmp_path / "run"

    # Lambdas are intentionally unpickleable in this context.
    env_cfg = {"callable": lambda x: x}
    train_cfg = module.TrainCfg()

    module._dump_run_configs(str(run_dir), env_cfg=env_cfg, train_cfg=train_cfg)

    assert (run_dir / "params" / "env.yaml").exists()
    assert (run_dir / "params" / "train.yaml").exists()
    assert not (run_dir / "params" / "env.pkl").exists()
    assert not (run_dir / "params" / "train.pkl").exists()


def test_load_train_cfg_rejects_non_positive_max_iterations():
    module = _load_train_irl_module()
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    with pytest.raises(ValueError, match="max_iterations"):
        module.load_train_cfg(
            task_name="Isaac-Lift-Cube-Franka-v0",
            args_cli=_args(max_iterations=0),
            config_dir=config_dir,
        )


def test_load_train_cfg_cli_overrides_irl_discount_gamma():
    module = _load_train_irl_module()
    config_dir = Path(__file__).resolve().parents[1] / "configs"

    cfg = module.load_train_cfg(
        task_name="Isaac-Lift-Cube-Franka-v0",
        args_cli=_args(irl_discount_gamma=0.7),
        config_dir=config_dir,
    )
    assert cfg.irl.discount_gamma == pytest.approx(0.7)
