from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from runner.runner import IrlRunner, IrlRunnerCfg
from utils.runtime_context import RuntimeContext
from storage.feature_storage import FeatureBufCfg


class _DummyEnv:
    def __init__(self, num_envs: int = 2, obs_dim: int = 4, action_dim: int = 3):
        self.num_envs = num_envs
        self.num_actions = action_dim
        self.obs_dim = obs_dim
        self.device = torch.device("cpu")
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.max_episode_length = 32

    def get_observations(self):
        return torch.zeros(self.num_envs, self.obs_dim, dtype=torch.float32)

    def step(self, actions: torch.Tensor):
        del actions
        obs_next = torch.ones(self.num_envs, self.obs_dim, dtype=torch.float32)
        rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        extras = {}
        return obs_next, rewards, dones, extras


class _DummyStorage:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


class _DummyRlAlg:
    def __init__(self, obs_dim: int = 4, action_dim: int = 3) -> None:
        self.policy = nn.Linear(obs_dim, action_dim)
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=1e-2)

        self.update_calls = 0
        self.process_calls = 0
        self.compute_returns_calls = 0

    def init_storage(self, *args, **kwargs) -> None:
        del args, kwargs

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        return torch.zeros(batch_size, self.policy.out_features, dtype=torch.float32)

    def process_env_step(self, obs_next: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        del obs_next, rewards, dones, extras
        self.process_calls += 1

    def compute_returns(self, obs: torch.Tensor) -> None:
        del obs
        self.compute_returns_calls += 1

    def update(self) -> None:
        self.update_calls += 1


class _DummyIrlAlg:
    def __init__(self, storage_size: int):
        self.reward = nn.Linear(2, 1)
        self.reward_optimizer = torch.optim.SGD(self.reward.parameters(), lr=1e-2)

        self._storage_size = storage_size
        self.reward_update_calls = 0
        self.process_env_step_calls = 0
        self.imitator_storage = None
        self.expert_storage = None
        self.env = None

    def init_imitator_storage(self, runtime_ctx: RuntimeContext, cfg: FeatureBufCfg) -> None:
        del runtime_ctx, cfg
        self.imitator_storage = _DummyStorage(self._storage_size)

    def init_expert_storage(self, runtime_ctx: RuntimeContext, cfg: FeatureBufCfg, num_envs: int) -> None:
        del runtime_ctx, cfg, num_envs
        self.expert_storage = _DummyStorage(self._storage_size)

    def clear_imitator_storage(self) -> None:
        # Keep storage length stable for deterministic unit tests.
        return

    def process_env_step(self, dones: torch.Tensor, features: torch.Tensor) -> None:
        del dones, features
        self.process_env_step_calls += 1

    def reward_update(self) -> tuple[float, float]:
        self.reward_update_calls += 1
        return 0.1, 0.2


def _runner_cfg() -> IrlRunnerCfg:
    return IrlRunnerCfg(
        num_steps_per_env_rl=4,
        save_interval=100,
        policy_updates_per_cycle=2,
        reward_updates_per_cycle=3,
        imitator_buffer=FeatureBufCfg(),
        expert_buffer=FeatureBufCfg(min_ep_len=1),
        expert_num_envs=1,
    )


def _feature_map(env: _DummyEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, 2, dtype=torch.float32)


def test_runner_respects_policy_and_reward_update_cycles():
    env = _DummyEnv()
    rl_alg = _DummyRlAlg()
    irl_alg = _DummyIrlAlg(storage_size=2)

    runner = IrlRunner(
        env=env,
        rl_alg=rl_alg,
        irl_alg=irl_alg,
        feature_map=_feature_map,
        runner_cfg=_runner_cfg(),
        runtime_ctx=RuntimeContext(num_envs=env.num_envs, feature_dim=2, device="cpu"),
        log_dir=None,
        device="cpu",
    )
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    assert rl_alg.update_calls == 2
    assert irl_alg.reward_update_calls == 3


def test_runner_skips_reward_updates_when_storage_is_empty():
    env = _DummyEnv()
    rl_alg = _DummyRlAlg()
    irl_alg = _DummyIrlAlg(storage_size=0)

    runner = IrlRunner(
        env=env,
        rl_alg=rl_alg,
        irl_alg=irl_alg,
        feature_map=_feature_map,
        runner_cfg=_runner_cfg(),
        runtime_ctx=RuntimeContext(num_envs=env.num_envs, feature_dim=2, device="cpu"),
        log_dir=None,
        device="cpu",
    )
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    assert rl_alg.update_calls == 2
    assert irl_alg.reward_update_calls == 0


def test_runner_checkpoint_payload_keys_and_load_roundtrip():
    env = _DummyEnv()
    rl_alg = _DummyRlAlg()
    irl_alg = _DummyIrlAlg(storage_size=0)

    runner = IrlRunner(
        env=env,
        rl_alg=rl_alg,
        irl_alg=irl_alg,
        feature_map=_feature_map,
        runner_cfg=_runner_cfg(),
        runtime_ctx=RuntimeContext(num_envs=env.num_envs, feature_dim=2, device="cpu"),
        log_dir=None,
        device="cpu",
    )
    runner.current_learning_iteration = 7

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "model.pt"
        runner.save(str(checkpoint_path))

        payload = torch.load(str(checkpoint_path), map_location="cpu")
        assert payload["iter"] == 7
        assert "model_state_dict" in payload
        assert "optimizer_state_dict" in payload
        assert "reward_model_state_dict" in payload
        assert "reward_optimizer_state_dict" in payload

        runner.current_learning_iteration = 0
        runner.load(str(checkpoint_path), load_optimizer=True)
        assert runner.current_learning_iteration == 7
