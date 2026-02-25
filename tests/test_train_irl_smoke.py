from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

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


class _FakeAppLauncher:
    def __init__(self, args) -> None:
        del args
        self.app = _FakeSimulationApp()

    @staticmethod
    def add_app_launcher_args(parser) -> None:
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--headless", action="store_true", default=True)


class _FakeObs(dict):
    def __init__(self) -> None:
        super().__init__({"policy": torch.zeros(1, 4), "critic": torch.zeros(1, 4)})

    def to(self, device):
        del device
        return self


class _FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.num_actions = 2
        self.device = torch.device("cpu")
        self.closed = False

    def get_observations(self):
        return _FakeObs()

    def step(self, actions):
        del actions
        obs_next = _FakeObs()
        rewards = torch.zeros(1, dtype=torch.float32)
        dones = torch.zeros(1, dtype=torch.bool)
        extras = {}
        return obs_next, rewards, dones, extras

    def seed(self, seed: int) -> None:
        del seed

    def close(self) -> None:
        self.closed = True


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

    def __getattr__(self, item):
        return getattr(self._env, item)


class _FakeActorCritic(nn.Module):
    def __init__(self, obs, obs_groups, num_actions: int, **kwargs):
        super().__init__()
        del obs, obs_groups, num_actions, kwargs
        self.linear = nn.Linear(1, 1)


class _FakePPO:
    def __init__(self, policy: nn.Module, device: str, **kwargs):
        del device, kwargs
        self.policy = policy
        self.optimizer = torch.optim.SGD(policy.parameters(), lr=1e-2)


class _FakeRewardModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        del cfg
        self.linear = nn.Linear(1, 1)


class _FakeIRL:
    def __init__(self, **kwargs) -> None:
        self.reward = kwargs["reward"]
        self.reward_optimizer = torch.optim.SGD(self.reward.parameters(), lr=1e-2)


class _FakeRunner:
    learn_calls = 0

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def load(self, path: str, load_optimizer: bool = True) -> None:
        del path, load_optimizer

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        del num_learning_iterations, init_at_random_ep_len
        _FakeRunner.learn_calls += 1


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_train_irl_main_smoke_with_mocked_runtime(tmp_path: Path, monkeypatch):
    module = _load_train_irl_module()

    config_root = tmp_path / "configs"
    _write_yaml(
        config_root / "train.yaml",
        {
            "defaults": {
                "env": "env/isaac_lift_cube_franka.yaml",
                "reward": "reward/dense_mlp.yaml",
                "policy": "policy/mlp.yaml",
                "algo": "algo/ppo.yaml",
            },
            "experiment_name": "smoke",
            "seed": 1,
            "max_iterations": 1,
            "runner": {
                "num_steps_per_env_rl": 1,
                "save_interval": 10,
                "policy_updates_per_cycle": 1,
                "reward_updates_per_cycle": 1,
                "expert_num_envs": 1,
            },
        },
    )
    _write_yaml(
        config_root / "env" / "isaac_lift_cube_franka.yaml",
        {"name": "Isaac-Lift-Cube-Franka-v0", "device": "cpu", "num_envs": 1},
    )
    _write_yaml(
        config_root / "reward" / "dense_mlp.yaml",
        {"type": "dense_mlp", "hidden_dims": [16], "is_linear": False, "activation": "elu"},
    )
    _write_yaml(
        config_root / "policy" / "mlp.yaml",
        {"actor_hidden_dims": [16], "critic_hidden_dims": [16], "activation": "elu"},
    )
    _write_yaml(
        config_root / "algo" / "ppo.yaml",
        {"learning_rate": 1.0e-4, "gamma": 0.98, "lam": 0.95},
    )

    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(module, "_build_feature_map", lambda cfg, device: (lambda env: torch.zeros(env.num_envs, 2)))
    monkeypatch.setattr(module, "_feature_dim_from_feature_map", lambda feature_map, env: 2)
    monkeypatch.setattr(module, "RewardModel", _FakeRewardModel)
    monkeypatch.setattr(module, "IRL", _FakeIRL)
    monkeypatch.setattr(module, "IrlRunner", _FakeRunner)

    fake_deps = module.RuntimeDeps(
        AppLauncher=_FakeAppLauncher,
        gym=_FakeGym,
        parse_env_cfg=lambda *args, **kwargs: {},
        get_checkpoint_path=lambda *args, **kwargs: "unused.pt",
        RslRlVecEnvWrapper=_FakeVecEnvWrapper,
        PPO=_FakePPO,
        ActorCritic=_FakeActorCritic,
        resolve_obs_groups=lambda obs, groups, default_sets=None: {"critic": []},
    )

    _FakeRunner.learn_calls = 0
    module.main(
        argv=["--task", "Isaac-Lift-Cube-Franka-v0", "--headless", "--device", "cpu"],
        deps=fake_deps,
    )
    assert _FakeRunner.learn_calls == 1


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
            ActorCritic=_FakeActorCritic,
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
        module.main(argv=["--task", "Isaac-Lift-Cube-Franka-v0", "--headless", "--device", "cpu"])

    assert call_order == ["load_app_launcher_cls", "app_launcher_init", "load_runtime_deps", "load_train_cfg"]
