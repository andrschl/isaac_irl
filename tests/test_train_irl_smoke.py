"""Smoke test for train_irl's runtime-deps init ordering.

The IsaacLab + rsl_rl import chain requires ``SimulationApp`` to be initialized
*before* those modules can be imported. ``train_irl.main`` enforces this by
loading the app launcher class first, instantiating it, and only then loading
the runtime-dep modules. This test guards that ordering — regressing it would
break every real training run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


def _load_train_irl_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "irl" / "train_irl.py"
    module_name = "train_irl_module_smoke_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeSimulationApp:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.num_actions = 2
        self.device = torch.device("cpu")


class _FakeGym:
    class wrappers:
        @staticmethod
        def RecordVideo(env, **kwargs):
            del kwargs
            return env

    @staticmethod
    def make(task_name: str, cfg, render_mode=None):
        del task_name, cfg, render_mode
        return _FakeEnv()


class _FakeVecEnvWrapper:
    def __init__(self, env: _FakeEnv) -> None:
        self._env = env
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.device = env.device


class _FakeMLPModel(nn.Module):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        super().__init__()
        del obs, obs_groups, obs_set, output_dim, kwargs
        self.linear = nn.Linear(1, 1)


class _FakeRolloutStorage:
    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",
    ):
        del training_type, num_envs, num_transitions_per_env, obs, actions_shape, device


class _FakePPO:
    def __init__(
        self, *, actor: nn.Module, critic: nn.Module, storage, device: str, **kwargs
    ):
        del storage, device, kwargs
        self.actor = actor
        self.critic = critic
        self.optimizer = torch.optim.SGD(
            list(actor.parameters()) + list(critic.parameters()), lr=1e-2
        )


def test_main_loads_runtime_deps_after_sim_app_init(monkeypatch):
    module = _load_train_irl_module()
    call_order: list[str] = []

    class _OrderAppLauncher:
        def __init__(self, args) -> None:
            del args
            call_order.append("app_launcher_init")
            self.app = _FakeSimulationApp()

        @staticmethod
        def add_app_launcher_args(parser) -> None:
            parser.add_argument("--device", type=str, default="cpu")
            parser.add_argument("--headless", action="store_true", default=True)

    def _fake_load_app_launcher_cls():
        call_order.append("load_app_launcher_cls")
        return _OrderAppLauncher

    def _fake_load_runtime_deps(app_launcher_cls):
        del app_launcher_cls
        call_order.append("load_runtime_deps")
        return module.RuntimeDeps(
            AppLauncher=_OrderAppLauncher,
            gym=_FakeGym,
            parse_env_cfg=lambda *args, **kwargs: {},
            get_checkpoint_path=lambda *args, **kwargs: "unused.pt",
            RslRlVecEnvWrapper=_FakeVecEnvWrapper,
            PPO=_FakePPO,
            MLPModel=_FakeMLPModel,
            RolloutStorage=_FakeRolloutStorage,
            resolve_obs_groups=lambda obs, groups, default_sets=None: {"critic": []},
        )

    def _fake_load_train_cfg(*, task_name, args_cli, config_dir=None):
        del task_name, args_cli, config_dir
        call_order.append("load_train_cfg")
        raise RuntimeError("stop-after-order-check")

    monkeypatch.setattr(module, "_load_app_launcher_cls", _fake_load_app_launcher_cls)
    monkeypatch.setattr(module, "_load_runtime_deps", _fake_load_runtime_deps)
    monkeypatch.setattr(module, "load_train_cfg", _fake_load_train_cfg)

    with pytest.raises(RuntimeError, match="stop-after-order-check"):
        module.main(
            argv=[
                "--task",
                "Isaac-Lift-Cube-Franka-v0",
                "--headless",
                "--device",
                "cpu",
            ]
        )

    assert call_order == [
        "load_app_launcher_cls",
        "app_launcher_init",
        "load_runtime_deps",
        "load_train_cfg",
    ]
