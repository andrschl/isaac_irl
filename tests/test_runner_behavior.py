from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from adapters import RslRlPpoAdapter
from interfaces import FeatureSpec, FeatureStep
from runner import IrlRunner, IrlRunnerCfg
from runner.loggers import NoopMetricLogger
from storage.feature_storage import FeatureBufCfg


class _DummyEnv:
    def __init__(self, num_envs: int = 2, obs_dim: int = 4, action_dim: int = 3):
        self.num_envs = num_envs
        self.num_actions = action_dim
        self.obs_dim = obs_dim
        self.device = torch.device("cpu")
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.max_episode_length = 32
        self.step_calls = 0
        self.last_actions: list[torch.Tensor] = []

    def get_observations(self):
        return torch.zeros(self.num_envs, self.obs_dim, dtype=torch.float32)

    def step(self, actions: torch.Tensor):
        self.step_calls += 1
        self.last_actions.append(actions.detach().clone())
        obs_next = torch.full((self.num_envs, self.obs_dim), float(self.step_calls), dtype=torch.float32)
        env_rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        extras: dict[str, Any] = {}
        return obs_next, env_rewards, dones, extras


class _NestedObsEnv(_DummyEnv):
    def get_observations(self):
        return {
            "policy": {
                "x": torch.zeros(self.num_envs, self.obs_dim, dtype=torch.float32)
            }
        }

    def step(self, actions: torch.Tensor):
        self.step_calls += 1
        self.last_actions.append(actions.detach().clone())
        obs_next = {
            "policy": {
                "x": torch.full(
                    (self.num_envs, self.obs_dim),
                    float(self.step_calls),
                    dtype=torch.float32,
                )
            }
        }
        env_rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        dones = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        extras: dict[str, Any] = {}
        return obs_next, env_rewards, dones, extras


def _obs_batch_size(obs: Any) -> int:
    if isinstance(obs, dict):
        return int(obs["policy"]["x"].shape[0])
    return int(obs.shape[0])


class _DummyReward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[10.0], [0.0]]))
        self.forward_calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return features @ self.weight


class _StubActor:
    """Minimal Actor stub used by _DummyRlAlg.

    ``act`` records "stochastic"; ``act_inference`` records "inference". The
    returned tensor's constant value differs per mode so tests can assert on
    which path was hit.
    """

    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim
        self.calls: list[str] = []

    def act(self, obs):
        self.calls.append("stochastic")
        return torch.full((_obs_batch_size(obs), self.action_dim), 1.0, dtype=torch.float32)

    def act_inference(self, obs):
        self.calls.append("inference")
        return torch.full((_obs_batch_size(obs), self.action_dim), 2.0, dtype=torch.float32)


class _DummyRlAlg:
    """Minimal RlAlgorithm stub matching the post-restructure contract.

    The runner only calls ``actor.act`` / ``actor.act_inference`` (validation
    + imitator rollouts), ``update()`` (no args), ``train_metrics`` /
    ``eval_metrics``, mode toggles, and save/load. RL-side env collection
    happens inside the real adapter's ``update()``; this stub just counts
    update() calls.
    """

    def __init__(self, action_dim: int = 3) -> None:
        self.action_dim = action_dim
        self.actor = _StubActor(action_dim)
        self.update_calls = 0
        self.train_mode_calls = 0
        self.eval_mode_calls = 0
        self.loaded_state: dict[str, Any] | None = None

    @property
    def act_modes(self) -> list[str]:
        """Test-only alias for the actor's call log."""
        return self.actor.calls

    def update(self) -> dict[str, float]:
        self.update_calls += 1
        return {"RL/policy_loss": float(self.update_calls)}

    def train_metrics(self) -> dict[str, float]:
        return {"RL/policy_std_mean": 0.5}

    def eval_metrics(self) -> dict[str, float]:
        return {"BC/val_loss": 0.25}

    def train_mode(self) -> None:
        self.train_mode_calls += 1

    def eval_mode(self) -> None:
        self.eval_mode_calls += 1

    def save_state(self) -> dict[str, Any]:
        return {"rl_alg_weight": torch.tensor([3.0])}

    def load_state(self, checkpoint: dict[str, Any], *, load_optimizer: bool = True) -> None:
        del load_optimizer
        self.loaded_state = checkpoint


class _DummyIrlAlg:
    def __init__(self, *, feature_dim: int = 2, has_data: bool = True) -> None:
        self.reward_model = _DummyReward()
        self.gamma = 1.0
        self.batch_size = 4
        self.feature_names = ["step", "env_idx"][:feature_dim]
        self.expert_success_rate = 0.75
        # Buffers are eagerly built by the real IrlAlgorithm; the stub
        # mirrors that with sentinel objects so anything that probes them
        # sees a non-None value.
        self.expert_storage = object()
        self.imitator_storage = object()
        self.has_data = has_data
        self.feature_steps: list[FeatureStep] = []
        self.clear_calls = 0
        self.finalize_calls = 0
        self.update_calls = 0
        self.loaded_state: dict[str, Any] | None = None

    def add_imitator(self, step: FeatureStep) -> None:
        self.feature_steps.append(step)

    def clear_imitator(self) -> None:
        self.clear_calls += 1
        self.feature_steps.clear()

    def finalize_imitator(self) -> None:
        self.finalize_calls += 1

    def can_update(self) -> bool:
        return self.has_data and len(self.feature_steps) > 0

    def update(self) -> dict[str, float]:
        self.update_calls += 1
        return {"IRL/reward_loss": 0.1, "IRL/feature_exp_diff_norm": 1.0}

    def per_feature_return_mean(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        return feats.mean(dim=(0, 1))

    def sample_expert_feature_return_mean(self, device: torch.device | str) -> torch.Tensor | None:
        return torch.ones(2, dtype=torch.float32, device=device)

    def train_metrics(self) -> dict[str, float]:
        return {"IRL/reward_param_norm": 1.0, "IRL/reward_param_max_abs": 10.0}

    def train_mode(self) -> None:
        self.reward_model.train()

    def eval_mode(self) -> None:
        self.reward_model.eval()

    def save_state(self) -> dict[str, Any]:
        return {"reward_weight": torch.tensor([4.0])}

    def load_state(self, checkpoint: dict[str, Any], *, load_optimizer: bool = True) -> None:
        del load_optimizer
        self.loaded_state = checkpoint


class _CaptureLogger(NoopMetricLogger):
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, float]]] = []
        self.finished: int | None = None

    def log(self, payload: dict[str, float], step: int) -> None:
        self.calls.append((step, dict(payload)))

    def finish(self, exit_code: int = 0) -> None:
        self.finished = exit_code


def _runner_cfg() -> IrlRunnerCfg:
    return IrlRunnerCfg(
        steps_per_env_per_cycle=2,
        save_interval=100,
        rl_updates_per_cycle=2,
        reward_updates_per_cycle=3,
        use_learned_reward=True,
        imitator_rollout_steps_per_env=3,
        validation_interval=0,
        imitator_buffer=FeatureBufCfg(min_ep_len=1),
        expert_buffer=FeatureBufCfg(min_ep_len=1),
        expert_num_envs=1,
    )


def _feature_map(env: _DummyEnv) -> torch.Tensor:
    step_feature = torch.full((env.num_envs,), float(env.step_calls), dtype=torch.float32)
    env_idx_feature = torch.arange(env.num_envs, dtype=torch.float32)
    return torch.stack((step_feature, env_idx_feature), dim=1)


class _StubFeatureSource:
    """FeatureSource that delegates to ``_feature_map``."""

    def __init__(self) -> None:
        self.spec = FeatureSpec(dim=2, names=("step", "env_idx"))

    def features(self, env) -> torch.Tensor:
        return _feature_map(env)


def _make_runner(*, use_learned_reward: bool = True, validation_interval: int = 0):
    env = _DummyEnv()
    rl_alg = _DummyRlAlg(action_dim=env.num_actions)
    irl_alg = _DummyIrlAlg(feature_dim=2)
    cfg = _runner_cfg()
    cfg.use_learned_reward = use_learned_reward
    cfg.validation_interval = validation_interval
    cfg.validation_steps_per_env = 2 if validation_interval > 0 else None
    logger = _CaptureLogger()
    runner = IrlRunner(
        env=env,
        rl_alg=rl_alg,
        irl_alg=irl_alg,
        feature_source=_StubFeatureSource(),
        runner_cfg=cfg,
        log_dir=None,
        device="cpu",
        metric_logger=logger,
    )
    return runner, env, rl_alg, irl_alg, logger


def test_runner_drives_rl_update_then_imitator_then_reward_update():
    """One learn cycle: N RL updates, one imitator rollout, M reward updates.

    The runner does not step the env for RL collection — that lives inside
    rl_alg.update(). env.step_calls therefore reflects only the imitator
    rollout, and rl_alg.update_calls reflects the configured RL update count.
    """
    runner, env, rl_alg, irl_alg, _ = _make_runner(use_learned_reward=True)

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    assert rl_alg.update_calls == runner.cfg.rl_updates_per_cycle
    assert rl_alg.act_modes == ["stochastic"] * runner.cfg.imitator_rollout_steps_per_env
    assert len(irl_alg.feature_steps) == runner.cfg.imitator_rollout_steps_per_env
    assert irl_alg.update_calls == runner.cfg.reward_updates_per_cycle
    assert env.step_calls == runner.cfg.imitator_rollout_steps_per_env


def test_runner_imitator_rollout_handles_nested_observation_trees():
    """Imitator collection through a nested-obs env populates the IRL feature buffer."""
    env = _NestedObsEnv()
    rl_alg = _DummyRlAlg(action_dim=env.num_actions)
    irl_alg = _DummyIrlAlg(feature_dim=2)
    cfg = _runner_cfg()
    cfg.use_learned_reward = False
    runner = IrlRunner(
        env=env,
        rl_alg=rl_alg,
        irl_alg=irl_alg,
        feature_source=_StubFeatureSource(),
        runner_cfg=cfg,
        log_dir=None,
        device="cpu",
        metric_logger=_CaptureLogger(),
    )

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    assert len(irl_alg.feature_steps) == cfg.imitator_rollout_steps_per_env
    assert irl_alg.feature_steps[0].dones.shape == (env.num_envs,)
    assert irl_alg.feature_steps[0].dones.dtype == torch.bool


def test_validation_uses_scratch_features_and_does_not_mutate_training_buffers():
    runner, _, rl_alg, irl_alg, logger = _make_runner(use_learned_reward=True, validation_interval=1)
    obs = runner._get_obs().to(runner.device)

    runner._run_validation(obs, iteration=0)

    assert irl_alg.feature_steps == []
    assert irl_alg.update_calls == 0
    assert rl_alg.act_modes == ["stochastic", "stochastic", "inference", "inference"]
    emitted = {key for _, payload in logger.calls for key in payload}
    assert "ValidationFeatures/expert/step" in emitted
    assert "ValidationFeatures/stochastic/imitator/step" in emitted
    assert "ValidationFeatures/inference/gap/env_idx" in emitted


def test_runner_checkpoint_payload_keys_and_load_roundtrip():
    runner, _, rl_alg, irl_alg, _ = _make_runner()
    runner.current_learning_iteration = 7
    runner.global_timestep = 123

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "model.pt"
        runner.save(str(checkpoint_path))

        payload = torch.load(str(checkpoint_path), map_location="cpu")
        assert payload["iter"] == 7
        assert payload["global_timestep"] == 123
        assert "rl_alg_state" in payload
        assert "irl_alg_state" in payload

        runner.current_learning_iteration = 0
        runner.global_timestep = 0
        runner.load(str(checkpoint_path), load_optimizer=True)
        assert runner.current_learning_iteration == 7
        assert runner.global_timestep == 123
        assert rl_alg.loaded_state is not None
        assert irl_alg.loaded_state is not None


def test_runner_finishes_injected_logger():
    runner, _, _, _, logger = _make_runner()

    runner.finish_logger(exit_code=3)

    assert logger.finished == 3


class _DummyMlpActor(nn.Module):
    """rsl_rl MLPModel-shaped stub: callable with stochastic_output flag."""

    def __init__(self, obs_dim: int = 4, action_dim: int = 3) -> None:
        super().__init__()
        self.mean = nn.Linear(obs_dim, action_dim)
        self.call_modes: list[bool] = []

    @property
    def out_features(self) -> int:
        return int(self.mean.out_features)

    def forward(self, obs: torch.Tensor, stochastic_output: bool = False) -> torch.Tensor:
        self.call_modes.append(bool(stochastic_output))
        return self.mean(obs) + (1.0 if stochastic_output else 0.0)


class _DummyPpo:
    """Minimal rsl_rl PPO-shaped stub for the adapter tests."""

    def __init__(self, obs_dim: int = 4, action_dim: int = 3) -> None:
        self.actor = _DummyMlpActor(obs_dim=obs_dim, action_dim=action_dim)
        self.collect_act_calls = 0
        self.process_calls = 0
        self.process_rewards: list[torch.Tensor] = []
        self.compute_returns_calls = 0
        self.update_calls = 0
        self.bc_only_update_calls = 0
        self.train_mode_calls = 0
        self.eval_mode_calls = 0
        self.loaded: dict[str, Any] | None = None

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self.collect_act_calls += 1
        return torch.zeros(obs.shape[0], self.actor.out_features)

    def process_env_step(self, obs_next, rewards, dones, extras) -> None:
        del obs_next, dones, extras
        self.process_calls += 1
        self.process_rewards.append(rewards.detach().clone())

    def compute_returns(self, obs) -> None:
        del obs
        self.compute_returns_calls += 1

    def update(self) -> dict[str, float]:
        self.update_calls += 1
        return {"surrogate": 0.25}

    def bc_only_update(self) -> dict[str, float]:
        self.bc_only_update_calls += 1
        return {"bc": 0.5, "bc_alpha": 1.0}

    def train_mode(self) -> None:
        self.train_mode_calls += 1

    def eval_mode(self) -> None:
        self.eval_mode_calls += 1

    def save(self) -> dict[str, torch.Tensor]:
        return {"actor_state_dict": torch.tensor([1.0])}

    def load(self, checkpoint: dict, load_cfg: dict, strict: bool) -> None:
        del load_cfg, strict
        self.loaded = checkpoint


class _DummyBCCfg:
    alpha = 1.0


class _DummyBCAlg:
    cfg = _DummyBCCfg()

    def compute_validation_loss(self, actor) -> float:
        del actor
        return 0.75


def _adapter_env(num_envs: int = 2, obs_dim: int = 4, action_dim: int = 3) -> _DummyEnv:
    return _DummyEnv(num_envs=num_envs, obs_dim=obs_dim, action_dim=action_dim)


def _identity_feature_map(env: _DummyEnv) -> torch.Tensor:
    """Return [N, 1] features whose value is the step counter."""
    return torch.full((env.num_envs, 1), float(env.step_calls), dtype=torch.float32)


class _ConstantRewardModel(nn.Module):
    def __init__(self, scale: float = 10.0) -> None:
        super().__init__()
        self.scale = float(scale)
        self.forward_calls = 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return features.squeeze(-1) * self.scale


def test_rsl_rl_adapter_update_drives_env_loop_and_passes_learned_rewards():
    """`adapter.update()` should step the env, score each step with the reward
    model, hand the learned rewards to `ppo.process_env_step`, finalize with
    `ppo.compute_returns`, then run `ppo.update`.
    """
    env = _adapter_env()
    ppo = _DummyPpo(obs_dim=env.obs_dim, action_dim=env.num_actions)
    reward_model = _ConstantRewardModel(scale=10.0)
    adapter = RslRlPpoAdapter(
        ppo,
        env=env,
        feature_map=_identity_feature_map,
        reward_model=reward_model,
        steps_per_env_per_cycle=3,
        use_learned_reward=True,
    )

    metrics = adapter.update()

    assert ppo.collect_act_calls == 3
    assert env.step_calls == 3
    assert ppo.process_calls == 3
    assert ppo.compute_returns_calls == 1
    assert ppo.update_calls == 1
    assert reward_model.forward_calls == 3
    assert metrics["RL/policy_loss"] == 0.25
    # First post-step features have step=1, reward = 10*step = 10. All envs same.
    assert torch.allclose(ppo.process_rewards[0], torch.full((env.num_envs,), 10.0))
    assert torch.allclose(ppo.process_rewards[2], torch.full((env.num_envs,), 30.0))


def test_rsl_rl_adapter_update_with_env_rewards_skips_reward_model():
    env = _adapter_env()
    ppo = _DummyPpo(obs_dim=env.obs_dim, action_dim=env.num_actions)
    reward_model = _ConstantRewardModel()
    adapter = RslRlPpoAdapter(
        ppo,
        env=env,
        feature_map=_identity_feature_map,
        reward_model=reward_model,
        steps_per_env_per_cycle=2,
        use_learned_reward=False,
    )

    adapter.update()

    assert reward_model.forward_calls == 0
    # All env rewards are zero from _DummyEnv.step.
    assert all(torch.allclose(r, torch.zeros(env.num_envs)) for r in ppo.process_rewards)


def test_rsl_rl_adapter_bc_only_skips_env_loop_and_reports_bc_metrics():
    env = _adapter_env()
    ppo = _DummyPpo(obs_dim=env.obs_dim, action_dim=env.num_actions)
    reward_model = _ConstantRewardModel()
    adapter = RslRlPpoAdapter(
        ppo,
        env=env,
        feature_map=_identity_feature_map,
        reward_model=reward_model,
        steps_per_env_per_cycle=2,
        use_learned_reward=True,
        bc_alg=_DummyBCAlg(),
    )

    metrics = adapter.update()
    eval_metrics = adapter.eval_metrics()
    train_metrics = adapter.train_metrics()

    assert ppo.collect_act_calls == 0
    assert env.step_calls == 0
    assert ppo.process_calls == 0
    assert ppo.compute_returns_calls == 0
    assert ppo.bc_only_update_calls == 1
    assert metrics["BC/loss"] == 0.5
    assert eval_metrics["BC/val_loss"] == 0.75
    assert train_metrics["BC/alpha"] == 1.0


def test_rsl_rl_adapter_checkpoint_delegates_to_wrapped_algorithm():
    env = _adapter_env()
    ppo = _DummyPpo(obs_dim=env.obs_dim, action_dim=env.num_actions)
    adapter = RslRlPpoAdapter(
        ppo,
        env=env,
        feature_map=_identity_feature_map,
        reward_model=_ConstantRewardModel(),
        steps_per_env_per_cycle=1,
    )

    state = adapter.save_state()
    adapter.load_state(state, load_optimizer=True)

    assert "actor_state_dict" in state
    assert ppo.loaded is state


def test_adapter_actor_act_does_not_invoke_ppo_collect_path():
    """`adapter.actor.act` must hit the pure actor, not `ppo.act`.

    Regression guard: rsl_rl PPO.act populates a pending Transition on the
    wrapped algorithm. Calling it during imitator/validation rollouts would
    silently affect PPO storage state. The actor facade routes stochastic
    actions through `actor(obs, stochastic_output=True)` instead.
    """
    env = _adapter_env()
    ppo = _DummyPpo(obs_dim=env.obs_dim, action_dim=env.num_actions)
    adapter = RslRlPpoAdapter(
        ppo,
        env=env,
        feature_map=_identity_feature_map,
        reward_model=_ConstantRewardModel(),
        steps_per_env_per_cycle=1,
    )

    adapter.actor.act(torch.zeros(2, 4))

    assert ppo.collect_act_calls == 0
    assert ppo.actor.call_modes == [True]
