from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from interfaces import EnvTransition, FeatureStep
from runner import IrlRunner, IrlRunnerCfg, RslRlPpoAdapter
from runner.loggers import NoopMetricLogger
from storage.feature_storage import FeatureBufCfg
from utils.runtime_context import RuntimeContext


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
    """Minimal RlAlgorithm stub for runner tests."""

    def __init__(self, action_dim: int = 3, *, collects: bool = True) -> None:
        self.action_dim = action_dim
        self.collects_rollouts = collects
        self.collect_action_calls = 0
        self.actor = _StubActor(action_dim)
        self.transitions: list[EnvTransition] = []
        self.end_rollout_calls = 0
        self.update_calls = 0
        self.train_mode_calls = 0
        self.eval_mode_calls = 0
        self.loaded_state: dict[str, Any] | None = None

    @property
    def act_modes(self) -> list[str]:
        """Test-only alias for the actor's call log."""
        return self.actor.calls

    def collect_action(self, obs: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        self.collect_action_calls += 1
        return torch.zeros((_obs_batch_size(obs), self.action_dim), dtype=torch.float32)

    def observe(self, transition: EnvTransition) -> None:
        self.transitions.append(transition)

    def end_rollout(self, last_obs) -> None:
        del last_obs
        self.end_rollout_calls += 1

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
        self.expert_storage = None
        self.imitator_storage = None
        self.has_data = has_data
        self.feature_steps: list[FeatureStep] = []
        self.clear_calls = 0
        self.finalize_calls = 0
        self.update_calls = 0
        self.loaded_state: dict[str, Any] | None = None

    def init_expert_storage(self, runtime_ctx: RuntimeContext, *, cfg: FeatureBufCfg | None = None, num_envs: int = 1) -> None:
        del runtime_ctx, cfg, num_envs
        self.expert_storage = object()

    def init_imitator_storage(self, runtime_ctx: RuntimeContext, *, cfg: FeatureBufCfg | None = None) -> None:
        del runtime_ctx, cfg
        self.imitator_storage = object()

    def observe(self, step: FeatureStep) -> None:
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
        feature_map=_feature_map,
        runner_cfg=cfg,
        runtime_ctx=RuntimeContext(num_envs=env.num_envs, feature_dim=2, device="cpu"),
        log_dir=None,
        device="cpu",
        metric_logger=logger,
    )
    return runner, env, rl_alg, irl_alg, logger


def test_runner_separates_rl_transitions_from_reward_feature_steps():
    runner, env, rl_alg, irl_alg, _ = _make_runner(use_learned_reward=True)

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    assert rl_alg.collect_action_calls == 2
    assert rl_alg.act_modes == ["stochastic"] * 3
    assert len(rl_alg.transitions) == 2
    assert all(not hasattr(transition, "features") for transition in rl_alg.transitions)
    assert rl_alg.end_rollout_calls == 1
    assert rl_alg.update_calls == runner.cfg.rl_updates_per_cycle
    assert len(irl_alg.feature_steps) == 3
    assert irl_alg.update_calls == runner.cfg.reward_updates_per_cycle
    assert env.step_calls == runner.cfg.steps_per_env_per_cycle + runner.cfg.imitator_rollout_steps_per_env


def test_runner_passes_learned_rewards_from_post_step_features_to_rl_alg():
    runner, _, rl_alg, irl_alg, _ = _make_runner(use_learned_reward=True)

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    first_transition = rl_alg.transitions[0]
    assert torch.allclose(first_transition.env_rewards, torch.zeros(2))
    # First post-step feature has step_feature=1, reward_model gives 10 * step.
    assert torch.allclose(first_transition.rewards, torch.full((2,), 10.0))
    assert irl_alg.reward_model.forward_calls == runner.cfg.steps_per_env_per_cycle


def test_runner_can_train_rl_alg_on_env_rewards_without_calling_reward_model_for_rl_rollout():
    runner, _, rl_alg, irl_alg, _ = _make_runner(use_learned_reward=False)

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    assert torch.allclose(rl_alg.transitions[0].rewards, rl_alg.transitions[0].env_rewards)
    assert irl_alg.reward_model.forward_calls == 0


def test_runner_handles_nested_observation_trees_and_done_vectors():
    env = _NestedObsEnv()
    rl_alg = _DummyRlAlg(action_dim=env.num_actions)
    irl_alg = _DummyIrlAlg(feature_dim=2)
    cfg = _runner_cfg()
    cfg.use_learned_reward = False
    runner = IrlRunner(
        env=env,
        rl_alg=rl_alg,
        irl_alg=irl_alg,
        feature_map=_feature_map,
        runner_cfg=cfg,
        runtime_ctx=RuntimeContext(num_envs=env.num_envs, feature_dim=2, device="cpu"),
        log_dir=None,
        device="cpu",
        metric_logger=_CaptureLogger(),
    )

    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

    first_transition = rl_alg.transitions[0]
    assert first_transition.obs["policy"]["x"].shape == (env.num_envs, env.obs_dim)
    assert first_transition.next_obs["policy"]["x"].device == runner.device
    assert first_transition.dones.shape == (env.num_envs,)
    assert first_transition.dones.dtype == torch.bool
    assert irl_alg.feature_steps[0].dones.shape == (env.num_envs,)


def test_validation_uses_scratch_features_and_does_not_mutate_training_buffers():
    runner, _, rl_alg, irl_alg, logger = _make_runner(use_learned_reward=True, validation_interval=1)
    obs = runner._get_obs().to(runner.device)

    runner._run_validation(obs, iteration=0)

    assert rl_alg.transitions == []
    assert irl_alg.feature_steps == []
    assert irl_alg.update_calls == 0
    # Validation must never hit the side-effectful collection path.
    assert rl_alg.collect_action_calls == 0
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


class _DummyActor(nn.Module):
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
        self.actor = _DummyActor(obs_dim=obs_dim, action_dim=action_dim)
        self.collect_act_calls = 0
        self.process_calls = 0
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
        del obs_next, rewards, dones, extras
        self.process_calls += 1

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


def test_rsl_rl_adapter_collect_vs_pure_action_paths():
    ppo = _DummyPpo()
    adapter = RslRlPpoAdapter(ppo)
    obs = torch.zeros(2, 4)

    _ = adapter.collect_action(obs)
    _ = adapter.actor.act(obs)
    _ = adapter.actor.act_inference(obs)
    transition = EnvTransition(
        obs=obs,
        actions=torch.zeros(2, 3),
        env_rewards=torch.zeros(2),
        rewards=torch.ones(2),
        dones=torch.zeros(2, dtype=torch.bool),
        next_obs=obs,
        extras={},
    )
    adapter.observe(transition)
    adapter.end_rollout(obs)
    metrics = adapter.update()

    assert ppo.collect_act_calls == 1
    # `act` should hit the pure actor (no side effects on PPO's pending transition).
    assert ppo.actor.call_modes == [True, False]
    assert ppo.process_calls == 1
    assert ppo.compute_returns_calls == 1
    assert metrics["RL/policy_loss"] == 0.25


def test_rsl_rl_adapter_bc_only_skips_collection_and_reports_bc_metrics():
    ppo = _DummyPpo()
    adapter = RslRlPpoAdapter(ppo, bc_alg=_DummyBCAlg())

    assert adapter.collects_rollouts is False
    metrics = adapter.update()
    eval_metrics = adapter.eval_metrics()
    train_metrics = adapter.train_metrics()

    assert ppo.bc_only_update_calls == 1
    assert metrics["BC/loss"] == 0.5
    assert eval_metrics["BC/val_loss"] == 0.75
    assert train_metrics["BC/alpha"] == 1.0


def test_rsl_rl_adapter_checkpoint_delegates_to_wrapped_algorithm():
    ppo = _DummyPpo()
    adapter = RslRlPpoAdapter(ppo)

    state = adapter.save_state()
    adapter.load_state(state, load_optimizer=True)

    assert "actor_state_dict" in state
    assert ppo.loaded is state


def test_adapter_actor_act_does_not_invoke_ppo_collect_path():
    """`adapter.actor.act` must hit the pure actor path, not `ppo.act`.

    Regression guard: rsl_rl PPO.act populates a pending Transition object on
    the wrapped algorithm. Calling it during imitator/validation rollouts would
    silently affect PPO storage state. The adapter must route stochastic
    actions through `actor(obs, stochastic_output=True)` instead.
    """
    ppo = _DummyPpo()
    adapter = RslRlPpoAdapter(ppo)

    adapter.actor.act(torch.zeros(2, 4))

    assert ppo.collect_act_calls == 0
    assert ppo.actor.call_modes == [True]


def test_adapter_bc_alpha_only_in_train_metrics_not_update_output():
    """BC/alpha is a constant; emitting it twice per iteration is noise."""
    ppo = _DummyPpo()
    adapter = RslRlPpoAdapter(ppo, bc_alg=_DummyBCAlg())

    update_metrics = adapter.update()
    train_metrics = adapter.train_metrics()

    assert "BC/alpha" not in update_metrics
    assert "BC/alpha" in train_metrics


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"steps_per_env_per_cycle": 0}, "steps_per_env_per_cycle"),
        ({"save_interval": 0}, "save_interval"),
        ({"rl_updates_per_cycle": 0}, "rl_updates_per_cycle"),
        ({"reward_updates_per_cycle": -1}, "reward_updates_per_cycle"),
        ({"expert_num_envs": 0}, "expert_num_envs"),
        ({"imitator_rollout_steps_per_env": 0}, "imitator_rollout_steps_per_env"),
        ({"validation_interval": -1}, "validation_interval"),
        ({"validation_steps_per_env": -1}, "validation_steps_per_env"),
    ],
)
def test_runner_cfg_rejects_invalid_values(overrides, match):
    cfg = _runner_cfg()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    with pytest.raises(ValueError, match=match):
        IrlRunner(
            env=_DummyEnv(),
            rl_alg=_DummyRlAlg(),
            irl_alg=_DummyIrlAlg(),
            feature_map=_feature_map,
            runner_cfg=cfg,
            runtime_ctx=RuntimeContext(num_envs=2, feature_dim=2, device="cpu"),
            log_dir=None,
            device="cpu",
            metric_logger=_CaptureLogger(),
        )
