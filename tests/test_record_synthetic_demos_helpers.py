from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")


class _DummyApp:
    def close(self) -> None:
        return


class _DummyAppLauncher:
    def __init__(self, args: argparse.Namespace) -> None:
        del args
        self.app = _DummyApp()

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--headless", action="store_true", default=True)


def _install_recording_import_stubs(monkeypatch) -> dict[str, Any]:
    state: dict[str, Any] = {"manager_calls": []}

    cli_args_module = types.ModuleType("cli_args")

    def _add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--dummy_rsl", type=int, default=0)

    cli_args_module.add_rsl_rl_args = _add_rsl_rl_args

    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_app_module = types.ModuleType("isaaclab.app")
    isaaclab_app_module.AppLauncher = _DummyAppLauncher
    isaaclab_utils_module = types.ModuleType("isaaclab.utils")
    isaaclab_utils_dict_module = types.ModuleType("isaaclab.utils.dict")
    isaaclab_utils_dict_module.print_dict = lambda *args, **kwargs: None
    isaaclab_module.app = isaaclab_app_module
    isaaclab_module.utils = isaaclab_utils_module
    isaaclab_utils_module.dict = isaaclab_utils_dict_module

    rsl_rl_module = types.ModuleType("rsl_rl")
    rsl_rl_runners_module = types.ModuleType("rsl_rl.runners")
    rsl_rl_runners_module.OnPolicyRunner = object
    rsl_rl_module.runners = rsl_rl_runners_module

    isaaclab_rl_module = types.ModuleType("isaaclab_rl")
    isaaclab_rl_rsl_rl_module = types.ModuleType("isaaclab_rl.rsl_rl")
    isaaclab_rl_rsl_rl_module.RslRlVecEnvWrapper = object
    isaaclab_rl_module.rsl_rl = isaaclab_rl_rsl_rl_module

    isaaclab_tasks_module = types.ModuleType("isaaclab_tasks")
    isaaclab_tasks_utils_module = types.ModuleType("isaaclab_tasks.utils")
    isaaclab_tasks_utils_module.get_checkpoint_path = lambda *args, **kwargs: "unused.pt"
    isaaclab_tasks_utils_module.parse_env_cfg = lambda *args, **kwargs: {}
    isaaclab_tasks_module.utils = isaaclab_tasks_utils_module

    gymnasium_module = types.ModuleType("gymnasium")

    collectors_module = types.ModuleType("collectors")
    collectors_module.RobomimicDataCollector = object

    reward_features_module = types.ModuleType("reward_features")
    reward_features_manager_based_module = types.ModuleType("reward_features.manager_based")

    def _manager_based_reward_feature_dict(*, env, ignored_reward_terms=(), device=None):
        state["manager_calls"].append(
            {
                "env": env,
                "ignored_reward_terms": set(ignored_reward_terms),
                "device": device,
            }
        )
        return {
            "term_a": torch.ones(2, dtype=torch.float32),
            "term_b": torch.zeros(2, dtype=torch.float32),
        }

    reward_features_manager_based_module.manager_based_reward_feature_dict = _manager_based_reward_feature_dict
    reward_features_module.manager_based = reward_features_manager_based_module

    monkeypatch.setitem(sys.modules, "cli_args", cli_args_module)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", isaaclab_utils_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.dict", isaaclab_utils_dict_module)
    monkeypatch.setitem(sys.modules, "rsl_rl", rsl_rl_module)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", rsl_rl_runners_module)
    monkeypatch.setitem(sys.modules, "isaaclab_rl", isaaclab_rl_module)
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl", isaaclab_rl_rsl_rl_module)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks", isaaclab_tasks_module)
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.utils", isaaclab_tasks_utils_module)
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium_module)
    monkeypatch.setitem(sys.modules, "collectors", collectors_module)
    monkeypatch.setitem(sys.modules, "reward_features", reward_features_module)
    monkeypatch.setitem(sys.modules, "reward_features.manager_based", reward_features_manager_based_module)
    return state


def _load_recording_module(monkeypatch, argv: list[str] | None = None):
    state = _install_recording_import_stubs(monkeypatch)
    argv = argv or ["record_synthetic_demos.py"]
    monkeypatch.setattr(sys, "argv", argv)

    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "recording" / "record_synthetic_demos.py"
    module_name = "record_synthetic_demos_test_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, state


def test_as_obs_mapping_accepts_mapping_and_wraps_tensor(monkeypatch):
    module, _ = _load_recording_module(monkeypatch)

    mapping = {"policy": torch.tensor([[1.0, 2.0]])}
    assert module._as_obs_mapping(mapping) is mapping

    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    wrapped = module._as_obs_mapping(tensor)
    assert set(wrapped.keys()) == {"policy"}
    assert torch.allclose(wrapped["policy"], tensor)


def test_get_policy_observations_prefers_wrapper_api(monkeypatch):
    module, _ = _load_recording_module(monkeypatch)

    class _EnvWithGetObs:
        def get_observations(self):
            return torch.tensor([[1.0, 2.0]]), {"ignored": True}

    policy_obs = module._get_policy_observations(_EnvWithGetObs())
    assert tuple(policy_obs.shape) == (1, 2)

    class _EnvWithResetOnly:
        def reset(self):
            return {"policy": torch.tensor([[3.0, 4.0]])}, {}

    reset_obs = module._get_policy_observations(_EnvWithResetOnly())
    assert "policy" in reset_obs


def test_step_env_parses_4tuple_and_5tuple(monkeypatch):
    module, _ = _load_recording_module(monkeypatch)

    class _Env4:
        def step(self, actions):
            del actions
            return torch.zeros(2, 3), torch.ones(2), torch.tensor([True, False]), {}

    class _Env5:
        def step(self, actions):
            del actions
            return torch.zeros(2, 3), torch.ones(2), torch.tensor([True, False]), torch.tensor([False, True]), {}

    _, r4, d4 = module._step_env(_Env4(), torch.zeros(2, 1))
    assert tuple(r4.shape) == (2,)
    assert tuple(d4.shape) == (2,)

    _, r5, d5 = module._step_env(_Env5(), torch.zeros(2, 1))
    assert tuple(r5.shape) == (2,)
    assert tuple(d5.shape) == (2,)
    assert d5.dtype == torch.bool


def test_import_rejects_custom_feature_type_fast(monkeypatch):
    with pytest.raises(NotImplementedError, match="custom is not implemented"):
        _load_recording_module(monkeypatch, ["record_synthetic_demos.py", "--feature_type", "custom"])


def test_main_rejects_non_default_feature_key(monkeypatch):
    module, _ = _load_recording_module(monkeypatch)
    module.args_cli.task = "Isaac-Unit-Test-v0"
    module.args_cli.feature_key = "alt_features"

    with pytest.raises(ValueError, match="--feature_key must be 'features'"):
        module.main()
