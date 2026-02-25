from __future__ import annotations

import argparse
import sys
import types

import pytest

pytest.importorskip("torch")

from scripts.rsl_rl import cli_args


class _Cfg:
    def __init__(self) -> None:
        self.seed = 0
        self.resume = False
        self.load_run = "old_run"
        self.load_checkpoint = "old.pt"
        self.run_name = "old_name"
        self.logger = "tensorboard"
        self.wandb_project = "old_wandb"
        self.neptune_project = "old_neptune"


def _namespace(**overrides) -> argparse.Namespace:
    payload = {
        "seed": None,
        "resume": None,
        "load_run": None,
        "checkpoint": None,
        "run_name": None,
        "logger": None,
        "log_project_name": None,
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_add_rsl_rl_args_parses_expected_flags():
    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)
    parsed = parser.parse_args(
        [
            "--experiment_name",
            "exp_1",
            "--run_name",
            "run_1",
            "--load_run",
            "run_prev",
            "--checkpoint",
            "model.pt",
            "--logger",
            "wandb",
            "--log_project_name",
            "proj_name",
        ]
    )

    assert parsed.experiment_name == "exp_1"
    assert parsed.run_name == "run_1"
    assert parsed.load_run == "run_prev"
    assert parsed.checkpoint == "model.pt"
    assert parsed.logger == "wandb"
    assert parsed.log_project_name == "proj_name"


def test_update_rsl_rl_cfg_overrides_requested_fields():
    cfg = _Cfg()
    args = _namespace(
        seed=123,
        resume=True,
        load_run="new_run",
        checkpoint="new_model.pt",
        run_name="fresh",
        logger="neptune",
        log_project_name="new_project",
    )

    out = cli_args.update_rsl_rl_cfg(cfg, args)
    assert out is cfg
    assert cfg.seed == 123
    assert cfg.resume is True
    assert cfg.load_run == "new_run"
    assert cfg.load_checkpoint == "new_model.pt"
    assert cfg.run_name == "fresh"
    assert cfg.logger == "neptune"
    assert cfg.wandb_project == "new_project"
    assert cfg.neptune_project == "new_project"


def test_update_rsl_rl_cfg_does_not_change_project_for_tensorboard():
    cfg = _Cfg()
    args = _namespace(logger="tensorboard", log_project_name="ignored_project")

    cli_args.update_rsl_rl_cfg(cfg, args)
    assert cfg.wandb_project == "old_wandb"
    assert cfg.neptune_project == "old_neptune"


def test_parse_rsl_rl_cfg_loads_registry_cfg_and_applies_overrides(monkeypatch):
    parse_cfg_mod = types.ModuleType("isaaclab_tasks.utils.parse_cfg")
    captured: list[tuple[str, str]] = []

    def _load_cfg_from_registry(task_name: str, key: str):
        captured.append((task_name, key))
        return _Cfg()

    parse_cfg_mod.load_cfg_from_registry = _load_cfg_from_registry
    utils_mod = types.ModuleType("isaaclab_tasks.utils")
    utils_mod.parse_cfg = parse_cfg_mod
    root_mod = types.ModuleType("isaaclab_tasks")
    root_mod.utils = utils_mod

    monkeypatch.setitem(sys.modules, "isaaclab_tasks", root_mod)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils.parse_cfg", parse_cfg_mod)

    args = _namespace(seed=7, run_name="from_cli", checkpoint="ckpt.pt")
    cfg = cli_args.parse_rsl_rl_cfg("Isaac-Lift-Cube-Franka-v0", args)

    assert captured == [("Isaac-Lift-Cube-Franka-v0", "rsl_rl_cfg_entry_point")]
    assert cfg.seed == 7
    assert cfg.run_name == "from_cli"
    assert cfg.load_checkpoint == "ckpt.pt"
