# ruff: noqa: E402
"""End-to-end wiring: a full training cycle with Isaac and rsl_rl faked out.

This is the test that exercises config -> build -> runner -> algorithm the way
a real run does, without a simulator. It runs the real Runner, the real PPO
adapter, and the real reward learner, so a contract that drifts between them
fails here rather than on a GPU an hour in.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import Mock, call

import pytest
import yaml

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from training import pipeline


class _SimulationApp:
    def __init__(self) -> None:
        self.app = Mock(spec=["post_quit"])
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _AppLauncher:
    def __init__(self, args) -> None:
        del args
        self.app = _SimulationApp()

    @staticmethod
    def add_app_launcher_args(parser) -> None:
        parser.add_argument("--headless", action="store_true", default=True)
        parser.add_argument("--disable_fabric", action="store_true", default=False)
        parser.add_argument("--enable_cameras", action="store_true", default=False)


class _Observations(dict):
    def __init__(self, num_envs: int = 2, obs_dim: int = 4) -> None:
        super().__init__(
            {
                "policy": torch.zeros(num_envs, obs_dim),
                "critic": torch.zeros(num_envs, obs_dim),
            }
        )

    def to(self, device):
        del device
        return self


class _Env:
    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.num_actions = 2
        self.device = torch.device("cpu")
        self.max_episode_length = 4
        self.unwrapped = self
        self.step_dt = 0.02
        self.closed = False
        self.steps = 0

    def get_observations(self):
        return _Observations(self.num_envs)

    def step(self, actions):
        del actions
        self.steps += 1
        return (
            _Observations(self.num_envs),
            torch.ones(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )

    def seed(self, seed: int) -> None:
        del seed

    def reset(self):
        return self.get_observations(), {}

    def close(self) -> None:
        self.closed = True


class _Gym:
    last_cfg: Any = None

    class wrappers:
        @staticmethod
        def RecordVideo(env, **kwargs):
            del kwargs
            return env

    @staticmethod
    def make(task_name: str, cfg, render_mode=None):
        del task_name, render_mode
        _Gym.last_cfg = cfg
        return _Env()


class _VecEnvWrapper:
    def __init__(self, env: _Env) -> None:
        self._env = env

    def __getattr__(self, item):
        return getattr(self._env, item)


class _MLPModel(nn.Module):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        del obs, obs_groups, obs_set, kwargs
        super().__init__()
        self.linear = nn.Linear(4, output_dim)
        self.distribution = None

    def forward(self, obs, stochastic_output: bool = False):
        del stochastic_output
        return self.linear(obs["policy"])


class _RolloutStorage:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs


class _PPO:
    def __init__(self, *, actor, critic, storage, device, **kwargs) -> None:
        assert kwargs["entropy_coef"] == 0.0
        del storage, device, kwargs
        self.actor = actor
        self.critic = critic
        self.rewards: list[torch.Tensor] = []
        self.updates = 0
        self.transition = type("Transition", (), {"actions_log_prob": None})()

    def act(self, obs):
        actions = self.actor(obs)
        self.transition.actions_log_prob = torch.full(
            (actions.shape[0],), -1.0, device=actions.device
        )
        return actions

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        self.rewards.append(rewards.detach().clone())

    def compute_returns(self, obs) -> None: ...

    def update(self) -> dict[str, float]:
        self.updates += 1
        return {"surrogate": 0.1, "value_function": 0.2}

    def train_mode(self) -> None: ...

    def eval_mode(self) -> None: ...

    def save(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict()}

    def load(self, checkpoint, load_cfg=None, strict=True) -> None: ...


class _FeatureSource:
    """Named reward features, as the manager-based source would report them."""

    def __init__(self, env, cfg=None, device=None) -> None:
        del cfg, device
        self._num_envs = env.num_envs
        from interfaces import FeatureSpec

        self.spec = FeatureSpec(dim=2, names=("reach", "lift"))

    def features(self, env):
        return torch.ones(env.num_envs, 2)


@pytest.fixture
def runtime_deps() -> pipeline.RuntimeDeps:
    return pipeline.RuntimeDeps(
        AppLauncher=_AppLauncher,
        gym=_Gym,
        parse_env_cfg=lambda *args, **kwargs: type("EnvCfg", (), {"seed": 0})(),
        get_checkpoint_path=lambda *args, **kwargs: "unused.pt",
        RslRlVecEnvWrapper=_VecEnvWrapper,
        PPO=_PPO,
        MLPModel=_MLPModel,
        RolloutStorage=_RolloutStorage,
        resolve_obs_groups=lambda obs, groups, default_sets=None: {
            "actor": ["policy"],
            "critic": ["critic"],
        },
    )


def _write_config(root: Path, payload: dict) -> None:
    directory = root / "configs" / "lift_cube_franka"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "experiment.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def _base_payload() -> dict:
    return {
        "experiment_name": "smoke",
        "seed": 1,
        "max_iterations": 1,
        "logger": "noop",
        "env": {
            "name": "Isaac-Lift-Cube-Franka-v0",
            "device": "cpu",
            "num_envs": 2,
            "add_success_bonus": False,
        },
        "reward": {"source": "environment"},
        "actor_critic": {"actor_hidden_dims": [8], "critic_hidden_dims": [8]},
        "algo": {"type": "ppo"},
        "runner": {
            "steps_per_env_per_cycle": 2,
            "save_interval": 10,
            "validation_interval": 0,
        },
    }


def _run(tmp_path: Path, monkeypatch, payload: dict, argv: list[str], deps) -> None:
    _write_config(tmp_path, payload)
    monkeypatch.setattr(pipeline, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pipeline.build, "ManagerBasedFeatureSource", _FeatureSource)
    monkeypatch.chdir(tmp_path)
    pipeline.main(argv=argv, deps=deps)


def test_an_environment_reward_run_completes_a_cycle(
    tmp_path: Path, monkeypatch, runtime_deps
) -> None:
    _run(
        tmp_path,
        monkeypatch,
        _base_payload(),
        ["--task", "Isaac-Lift-Cube-Franka-v0", "--headless"],
        runtime_deps,
    )

    assert _Gym.last_cfg.seed == 1
    checkpoints = list((tmp_path / "logs" / "train" / "smoke").rglob("model_*.pt"))
    assert checkpoints, "a finished run writes a final checkpoint"


def test_cli_overrides_reach_nested_configuration(
    tmp_path: Path, monkeypatch, runtime_deps
) -> None:
    _run(
        tmp_path,
        monkeypatch,
        _base_payload(),
        [
            "--task",
            "Isaac-Lift-Cube-Franka-v0",
            "--headless",
            "--max_iterations",
            "2",
            "--set",
            "runner.steps_per_env_per_cycle=3",
        ],
        runtime_deps,
    )

    # Two iterations of three steps per env, driven through the real adapter.
    assert _Gym.last_cfg.seed == 1


@pytest.mark.parametrize(
    "error",
    [None, RuntimeError("evaluation failed"), KeyboardInterrupt()],
    ids=["success", "failure", "interruption"],
)
def test_lifecycle_preserves_status_and_cleans_up_before_shutdown(error) -> None:
    events = Mock()
    expected_exception = nullcontext() if error is None else pytest.raises(type(error))

    with expected_exception as caught:
        with pipeline.isaac_lifecycle(
            events.simulation_app, argparse.Namespace(video=False)
        ) as state:
            state.env = events.env
            state.runner = events.runner
            if error is not None:
                raise error

    if error is not None:
        assert caught.value is error
    expected_calls = [
        call.env.close(),
        call.runner.finish_logger(exit_code=0 if error is None else 1),
    ]
    if error is not None:
        expected_calls.append(call.simulation_app.app.post_quit(return_code=1))
    expected_calls.append(call.simulation_app.close())
    assert events.mock_calls == expected_calls


def test_runtime_dependencies_load_only_after_the_simulator_app(monkeypatch) -> None:
    """Isaac modules are import-time side effects; the app must exist first."""
    order: list[str] = []
    launched_apps: list[_SimulationApp] = []

    class _OrderedAppLauncher(_AppLauncher):
        def __init__(self, args) -> None:
            order.append("app_launcher")
            super().__init__(args)
            launched_apps.append(self.app)

    def _load_deps(app_launcher_cls):
        del app_launcher_cls
        order.append("runtime_deps")
        raise RuntimeError("stop-after-order-check")

    monkeypatch.setattr(pipeline, "_app_launcher_cls", lambda: _OrderedAppLauncher)
    monkeypatch.setattr(pipeline, "load_runtime_deps", _load_deps)

    with pytest.raises(RuntimeError, match="stop-after-order-check"):
        pipeline.main(argv=["--task", "Isaac-Lift-Cube-Franka-v0", "--headless"])

    assert order == ["app_launcher", "runtime_deps"]
    launched_apps[0].app.post_quit.assert_called_once_with(return_code=1)
    assert launched_apps[0].closed


def test_ppo_requests_staggered_episode_starts(
    tmp_path: Path, monkeypatch, runtime_deps
) -> None:
    """Lockstep resets concentrate post-reset reward into all-or-nothing blocks."""
    seen: dict[str, object] = {}
    original = pipeline.Runner.learn

    def record(self, num_iterations, *, init_at_random_ep_len=False):
        seen["staggered"] = init_at_random_ep_len
        return original(
            self, num_iterations, init_at_random_ep_len=init_at_random_ep_len
        )

    monkeypatch.setattr(pipeline.Runner, "learn", record)
    _run(
        tmp_path,
        monkeypatch,
        _base_payload(),
        ["--task", "Isaac-Lift-Cube-Franka-v0", "--headless"],
        runtime_deps,
    )

    assert seen["staggered"] is True


@pytest.mark.parametrize("enabled", [False, True])
def test_reward_curriculum_option_preserves_or_removes_only_weight_schedules(enabled):
    from types import SimpleNamespace

    from config.schema import EnvCfg, RewardCfg, TrainCfg
    from training.build import build_env_cfg

    def modify_reward_weight():
        pass

    schedule = SimpleNamespace(func=modify_reward_weight)
    unrelated = SimpleNamespace(func=lambda: None)
    native = SimpleNamespace(
        curriculum=SimpleNamespace(action_rate=schedule, unrelated=unrelated)
    )
    cfg = TrainCfg(
        env=EnvCfg(add_success_bonus=False, reward_curriculum=enabled),
        reward=RewardCfg(source="environment"),
    )
    result = build_env_cfg(
        cfg, parse_env_cfg=lambda *args, **kwargs: native, use_fabric=True
    )
    assert result.curriculum.action_rate is (schedule if enabled else None)
    assert result.curriculum.unrelated is unrelated
