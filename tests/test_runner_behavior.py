"""The runner's cycle, validation contract, and the PPO adapter that feeds it."""

# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from adapters import RslRlPpoAdapter
from config.schema import FeatureBufCfg, RunnerCfg
from interfaces import FeatureSpec, Generation
from reward.features import FeatureColumns
from runner import Runner
from runner.loggers import LogEvent, NoopLogger


class _Env:
    """Vectorized env stub: constant observations, counted steps."""

    resets = 0

    def __init__(self, num_envs: int = 2, obs_dim: int = 4, reward: float = 1.0):
        self.resets = 0
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = torch.device("cpu")
        self.max_episode_length = 8
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.reward = reward
        self.step_calls = 0
        self.done_every: int | None = None

    def get_observations(self):
        return torch.zeros(self.num_envs, self.obs_dim)

    def reset(self):
        self.resets += 1
        return self.get_observations(), {}

    def step(self, actions: torch.Tensor):
        self.step_calls += 1
        done = self.done_every is not None and self.step_calls % self.done_every == 0
        return (
            torch.full((self.num_envs, self.obs_dim), float(self.step_calls)),
            torch.full((self.num_envs,), self.reward),
            torch.full((self.num_envs,), done, dtype=torch.bool),
            {},
        )


class _FeatureSource:
    def __init__(self, num_envs: int = 2, dim: int = 3):
        self.spec = FeatureSpec(dim=dim, names=tuple(f"term_{i}" for i in range(dim)))
        self._num_envs = num_envs

    def features(self, env: Any) -> torch.Tensor:
        del env
        return torch.ones(self._num_envs, self.spec.dim)


class _Actor:
    def __init__(self, num_envs: int = 2, action_dim: int = 3):
        self._shape = (num_envs, action_dim)
        self.inference_calls = 0
        self.stochastic_calls = 0

    def act(self, obs: Any) -> torch.Tensor:
        del obs
        self.stochastic_calls += 1
        return torch.zeros(self._shape)

    def act_inference(self, obs: Any) -> torch.Tensor:
        del obs
        self.inference_calls += 1
        return torch.ones(self._shape)


class _RlAlg:
    """Algorithm stub recording the order the runner drives it in."""

    def __init__(self, num_envs: int = 2, horizon: int = 4, feature_dim: int = 3):
        self.actor = _Actor(num_envs)
        self.calls: list[str] = []
        self.transition_count = num_envs * horizon
        # One distinguishable constant per column, so a column selection that
        # picks the wrong ones is visible in the value, not only in the shape.
        self._features = torch.arange(float(feature_dim)).expand(
            num_envs, horizon, feature_dim
        )
        self._dones = torch.zeros(num_envs, horizon, dtype=torch.bool)
        self.collection_resets = 0
        self.policy_version = 7

    @property
    def simulator_steps_per_update(self) -> int:
        return self.transition_count

    def collect(self) -> Generation:
        self.calls.append("collect")
        return Generation(
            transition_count=self.transition_count,
            features=self._features,
            dones=self._dones,
            policy_version=self.policy_version,
        )

    def update(self) -> dict[str, float]:
        self.calls.append("update")
        return {"DirectQ/td_loss": 0.5, "Value/q_mean": 2.0}

    def reset_collection_state(self) -> None:
        self.collection_resets += 1

    def train_metrics(self) -> dict[str, float]:
        return {"Actor/policy_std_mean": 0.3}

    def eval_metrics(self) -> dict[str, float]:
        return {}

    def train_mode(self) -> None:
        self.calls.append("train_mode")

    def eval_mode(self) -> None:
        self.calls.append("eval_mode")

    def save_state(self) -> dict[str, Any]:
        return {"policy": 1}

    def load_state(self, checkpoint: dict[str, Any], *, load_optimizer: bool = True):
        self.calls.append("load_state")


class _RewardLearner:
    """Reward-learner stub with the surface the runner actually uses."""

    def __init__(self, feature_dim: int = 3):
        self.calls: list[str] = []
        self.replaced: list[tuple[int, ...]] = []
        self.received: list[torch.Tensor] = []
        self.expert_success_rate = 0.75
        self.gamma = 0.99
        self.batch_size = 4
        self._feature_dim = feature_dim

    def replace_imitator_trajectories(self, features, valid) -> None:
        self.calls.append("replace_imitator")
        self.replaced.append(tuple(features.shape))
        self.received.append(features)

    def replace_imitator_episodes(self, features, mask) -> None:
        self.calls.append("replace_episodes")
        self.replaced.append(tuple(features.shape))
        self.received.append(features)

    def can_update(self) -> bool:
        return True

    def update(self) -> dict[str, float]:
        self.calls.append("reward_update")
        return {
            "IRL/reward_loss": -1.0,
            "IRL/feature_exp_diff_norm": 0.25,
            "IRL/reward_optimizer_steps": 8.0,
        }

    def train_metrics(self) -> dict[str, float]:
        return {"IRL/reward_param_norm": 1.5}

    def per_feature_return_mean(self, feats, mask) -> torch.Tensor:
        return torch.zeros(self._feature_dim)

    def sample_expert_feature_return_mean(self, device) -> torch.Tensor:
        return torch.ones(self._feature_dim)

    def train_mode(self) -> None: ...

    def eval_mode(self) -> None: ...

    def save_state(self) -> dict[str, Any]:
        return {"reward": 2}

    def load_state(self, checkpoint, *, load_optimizer: bool = True) -> None: ...


class _CaptureLogger(NoopLogger):
    def __init__(self) -> None:
        self.events: list[LogEvent] = []
        self.finished_with: int | None = None

    def log(self, event: LogEvent) -> None:
        self.events.append(event)

    def finish(self, exit_code: int = 0) -> None:
        self.finished_with = exit_code

    def payloads(self, kind: str) -> list[dict[str, float]]:
        return [event.payload for event in self.events if event.kind == kind]


def _runner(
    *, reward_features: FeatureColumns | None = None, **cfg_overrides
) -> tuple[Runner, _RlAlg, _RewardLearner, _CaptureLogger]:
    env = _Env()
    rl_alg = _RlAlg()
    feature_source = _FeatureSource()
    reward_learner = _RewardLearner(
        feature_dim=feature_source.spec.dim
        if reward_features is None
        else reward_features.dim
    )
    logger = _CaptureLogger()
    settings: dict[str, Any] = {
        "steps_per_env_per_cycle": 4,
        "validation_interval": 0,
        "imitator_buffer": FeatureBufCfg(min_ep_len=1),
    }
    settings.update(cfg_overrides)
    cfg = RunnerCfg(**settings)
    runner = Runner(
        env,
        rl_alg=rl_alg,
        reward_learner=reward_learner,
        feature_source=feature_source,
        cfg=cfg,
        logger=logger,
        reward_features=reward_features,
    )
    return runner, rl_alg, reward_learner, logger


def test_reward_is_updated_between_collection_and_policy_improvement() -> None:
    """The policy must never improve on a reward the new data has not seen."""
    runner, rl_alg, reward_learner, _ = _runner()

    runner.learn(1)

    assert rl_alg.calls == ["train_mode", "collect", "update"]
    assert reward_learner.calls == ["replace_imitator", "reward_update"]


def test_collected_features_are_reused_instead_of_a_second_rollout() -> None:
    runner, rl_alg, reward_learner, _ = _runner(imitator_source="collected")

    runner.learn(1)

    assert reward_learner.replaced == [(2, 4, 3)]
    assert runner.env.step_calls == 0, "collected features cost no extra env steps"


def test_the_reward_learner_only_sees_the_columns_it_fits() -> None:
    """Replay carries every tapped term; the reward learner fits a subset.

    The sparse terminal-success coordinate is part of the reward the policy
    maximizes, so the critics have to see it. Handing it to IRL as well would
    ask the reward learner to match the expert on the one thing the task
    already pays for outright, and its coefficient is not learned anyway.
    """
    columns = FeatureColumns.select(
        ("term_0", "term_1", "term_2"), ("term_2", "term_0")
    )
    runner, _, reward_learner, logger = _runner(
        reward_features=columns, validation_interval=1, validation_steps_per_env=2
    )

    runner.learn(1)

    assert reward_learner.replaced == [(2, 4, 2)]
    # Columns 2 and 0 of [0, 1, 2], in the order the reward learner asked for.
    torch.testing.assert_close(
        reward_learner.received[0][0, 0], torch.tensor([2.0, 0.0])
    )
    detail = logger.payloads("validation/features")[0]
    assert "ValidationFeatures/inference/gap/term_2" in detail
    assert "ValidationFeatures/inference/gap/term_1" not in detail


def test_reward_updates_report_their_real_cost_and_whose_data_they_fit() -> None:
    """Every optimizer step is counted, and against the policy that collected.

    The feature expectations being matched belong to the frozen behaviour
    policy of this batch. More reward steps fit that fixed measured gap more
    exactly; they do not make the gap smaller, so reading the two together is
    the only way the number means anything.
    """
    runner, rl_alg, reward_learner, logger = _runner(reward_updates_per_cycle=3)

    runner.learn(1)

    payload = logger.payloads("iteration_summary")[0]
    assert reward_learner.calls.count("reward_update") == 3
    assert payload["IRL/reward_optimizer_steps"] == 24.0
    assert payload["IRL/reward_updates"] == 3.0
    assert payload["IRL/collection_policy_version"] == float(rl_alg.policy_version)


def test_a_dedicated_imitator_rollout_collects_whole_episodes() -> None:
    runner, rl_alg, reward_learner, _ = _runner(imitator_source="rollout")

    runner.learn(1)

    # One row per env, one column per env step, over the task's episode length.
    assert reward_learner.replaced == [(2, 8, 3)]
    assert runner.env.step_calls == 8


def test_the_transition_budget_stops_the_run_before_the_iteration_count() -> None:
    runner, rl_alg, _, _ = _runner(transition_budget=16)

    runner.learn(100)

    assert runner.current_iteration == 2
    assert runner.global_timestep == 16


def test_every_iteration_reports_the_summary_contract() -> None:
    runner, _, _, logger = _runner()

    runner.learn(1)

    payload = logger.payloads("iteration_summary")[0]
    assert payload["Summary/critic_loss"] == 0.5
    assert payload["Summary/q_mean"] == 2.0
    assert payload["Summary/policy_std"] == 0.3
    assert payload["Summary/reward_loss"] == -1.0
    assert payload["Summary/feature_gap_norm"] == 0.25
    assert payload["Summary/transitions"] == 8.0


def test_validation_scores_both_policy_modes_and_returns_to_training() -> None:
    runner, rl_alg, _, logger = _runner(
        validation_interval=1, validation_steps_per_env=2
    )

    runner.learn(1)

    payload = logger.payloads("validation/summary")[0]
    assert payload["Validation/return_mean"] == pytest.approx(2.0)
    assert "Validation/stochastic/return_mean" in payload
    assert payload["Summary/expert_success_rate"] == 0.75
    assert rl_alg.actor.inference_calls == 2
    assert rl_alg.actor.stochastic_calls == 2
    assert rl_alg.calls[-1] == "train_mode", "training resumes after validation"


def test_validation_reports_the_gap_to_the_expert_per_feature() -> None:
    runner, _, _, logger = _runner(validation_interval=1, validation_steps_per_env=2)

    runner.learn(1)

    detail = logger.payloads("validation/features")[0]
    assert detail["ValidationFeatures/inference/gap/term_0"] == 1.0
    assert detail["ValidationFeatures/inference/imitator/term_0"] == 0.0


def test_checkpoints_round_trip_both_learners(tmp_path: Path) -> None:
    runner, rl_alg, _, _ = _runner()
    runner.learn(1)
    path = str(tmp_path / "model.pt")

    runner.save(path)
    runner.current_iteration = 0
    runner.global_timestep = 0
    runner.load(path)

    payload = torch.load(path, weights_only=False)
    assert set(payload) == {
        "iteration",
        "global_timestep",
        "rl_alg_state",
        "reward_learner_state",
        "interaction_counts",
    }
    assert runner.current_iteration == 1
    assert "load_state" in rl_alg.calls


def test_complete_collected_episodes_replace_irl_data_without_extra_interaction():
    runner, alg, reward, _ = _runner()
    generation = Generation(
        transition_count=6,
        features=torch.ones(2, 4, 3),
        episode_mask=torch.tensor([[True, True, False, False], [True] * 4]),
        environment_transition_count=8,
        completed_episodes=2,
    )
    alg.collect = lambda: generation
    runner.learn(1)
    assert reward.calls == ["replace_episodes", "reward_update"]
    assert runner.env.step_calls == 0
    assert runner.global_timestep == 6
    assert runner.environment_transitions == 8
    assert runner.training_episodes == 2


def test_total_budget_reserves_a_whole_training_batch():
    runner, alg, _, _ = _runner(total_transition_budget=15)
    runner.learn(5)
    assert alg.calls.count("collect") == 1
    assert runner.environment_transitions == 8


def test_evaluation_is_skipped_when_both_modes_do_not_fit():
    runner, _, _, logger = _runner(
        total_transition_budget=15, validation_interval=1, validation_steps_per_env=4
    )
    runner.learn(1)
    assert runner.evaluation_transitions == 0
    assert logger.payloads("validation/skipped")


def test_final_video_cannot_exceed_the_interaction_budget():
    runner, _, _, logger = _runner(total_transition_budget=15)
    runner.learn(1)
    # The recorder must not even be touched when the clips cannot fit.
    runner.record_final_rollouts(recorder=None, video_length=4)
    assert runner.environment_transitions == 8
    assert logger.payloads("video/skipped")


def test_old_checkpoint_cannot_resume_an_unknown_interaction_budget(tmp_path):
    runner, _, _, _ = _runner()
    path = tmp_path / "old.pt"
    torch.save({"iteration": 1, "global_timestep": 8}, path)
    with pytest.raises(ValueError, match="interaction_counts"):
        runner.load(str(path))


def test_episode_validation_does_not_step_reset_or_stagger_the_runner_env():
    from conftest import build_rollout

    from storage.episode_batch import EpisodeBatch

    runner, alg, _, logger = _runner(validation_interval=1, validation_episodes=2)
    runner.env.episode_length_buf.fill_(3)
    observed_modes = []

    def evaluate(mode, episodes):
        assert alg.calls.index("update") < alg.calls.index("eval_mode")
        assert episodes == 2
        observed_modes.append(mode)
        rollout = build_rollout(num_envs=2, horizon=4, feature_dim=3)
        rollout.truncated[:, -1] = True
        return EpisodeBatch(rollout, torch.ones(2, 4, dtype=torch.bool), 8)

    runner.episode_evaluator = evaluate
    runner.learn(1, init_at_random_ep_len=True)
    assert observed_modes == ["inference", "stochastic"]
    assert runner.env.resets == runner.env.step_calls == 0
    assert (runner.env.episode_length_buf == 3).all()
    assert runner.environment_transitions == 24
    assert runner.evaluation_transitions == 16
    assert runner.evaluation_episodes == 4
    assert (
        logger.payloads("validation/summary")[-1]["Summary/environment_transitions"]
        == 24
    )


def test_a_run_without_a_reward_learner_skips_reward_updates() -> None:
    env = _Env()
    rl_alg = _RlAlg()
    runner = Runner(
        env,
        rl_alg=rl_alg,
        reward_learner=None,
        feature_source=_FeatureSource(),
        cfg=RunnerCfg(steps_per_env_per_cycle=4, validation_interval=0),
        logger=_CaptureLogger(),
    )

    runner.learn(1)

    assert rl_alg.calls == ["train_mode", "collect", "update"]


def test_episode_validation_reports_terminal_success():
    from conftest import build_rollout

    from storage.episode_batch import EpisodeBatch

    runner, _, _, _ = _runner(validation_episodes=2)
    runner.feature_source.spec = FeatureSpec(1, ("terminal_success",))
    rollout = build_rollout(num_envs=2, horizon=4, feature_dim=1)
    rollout.features.zero_()
    rollout.features[0, -1, 0] = 50.0
    rollout.terminated.zero_()
    rollout.truncated.zero_()
    rollout.truncated[:, -1] = True
    batch = EpisodeBatch(rollout, torch.ones(2, 4, dtype=torch.bool), 8)
    runner.episode_evaluator = lambda mode, count: batch
    result = runner._episode_validation("inference")
    assert result.ever_success.tolist() == [True, False]


def test_collected_imitator_features_are_required_when_the_algorithm_has_none() -> None:
    runner, rl_alg, _, _ = _runner(imitator_source="collected")
    rl_alg._features = None
    rl_alg._dones = None

    with pytest.raises(RuntimeError, match="imitator_source='collected'"):
        runner.learn(1)


# --------------------------------------------------------------- PPO adapter


class _MlpActor(nn.Module):
    def __init__(self, action_dim: int = 3):
        super().__init__()
        self.action_dim = action_dim
        self.distribution = None
        self.stochastic_calls = 0

    def forward(self, obs, stochastic_output: bool = False):
        self.stochastic_calls += int(stochastic_output)
        return torch.zeros(obs.shape[0], self.action_dim)


class _Ppo:
    def __init__(self) -> None:
        self.actor = _MlpActor()
        self.rewards: list[torch.Tensor] = []
        self.updates = 0
        self.bc_only_updates = 0
        self.returns_computed = 0
        self.sample_log_probability = -2.0
        self.transition = type("Transition", (), {"actions_log_prob": None})()

    def act(self, obs):
        self.transition.actions_log_prob = torch.full(
            (obs.shape[0],), self.sample_log_probability
        )
        return torch.zeros(obs.shape[0], self.actor.action_dim)

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        self.rewards.append(rewards.detach().clone())

    def compute_returns(self, obs) -> None:
        self.returns_computed += 1

    def update(self) -> dict[str, float]:
        self.updates += 1
        return {"surrogate": 0.25, "value_function": 0.5}

    def bc_only_update(self) -> dict[str, float]:
        self.bc_only_updates += 1
        return {"surrogate": 0.0, "bc": 0.75}

    def train_mode(self) -> None: ...

    def eval_mode(self) -> None: ...


class _Reward:
    feature_dim = 3

    def score(self, features: torch.Tensor) -> torch.Tensor:
        return features.sum(dim=-1) * 2.0


class _BcCfg:
    weight = 1.0
    loss_type = "nll"


class _Bc:
    cfg = _BcCfg()

    def compute_validation_loss(self, actor) -> float:
        return 0.125


def _ppo_adapter(**overrides) -> tuple[RslRlPpoAdapter, _Ppo, _Env]:
    env = _Env()
    ppo = _Ppo()
    kwargs = dict(
        env=env,
        feature_source=_FeatureSource(),
        reward=_Reward(),
        steps_per_env_per_cycle=3,
    )
    kwargs.update(overrides)
    return RslRlPpoAdapter(ppo, **kwargs), ppo, env


def test_ppo_collect_scores_every_step_with_the_learned_reward() -> None:
    adapter, ppo, env = _ppo_adapter()

    generation = adapter.collect()

    assert env.step_calls == 3
    assert ppo.returns_computed == 1
    assert generation.transition_count == 6
    assert tuple(generation.features.shape) == (2, 3, 3)
    # Three feature terms, each 1.0, scored at 2x.
    assert torch.allclose(ppo.rewards[0], torch.full((2,), 6.0))


def test_ppo_collect_passes_the_environment_reward_through_unchanged() -> None:
    adapter, ppo, env = _ppo_adapter(reward=None, feature_source=None)

    generation = adapter.collect()

    assert torch.allclose(ppo.rewards[0], torch.full((2,), env.reward))
    assert generation.features is None
    assert adapter.train_metrics()["Train/reward_per_step"] == env.reward


def test_ppo_entropy_rate_is_integrated_over_the_environment_step() -> None:
    adapter, ppo, env = _ppo_adapter(
        reward=None,
        feature_source=None,
        entropy_rate=0.3,
        step_dt=0.02,
    )

    adapter.collect()

    expected_bonus = -0.02 * 0.3 * ppo.sample_log_probability
    torch.testing.assert_close(
        ppo.rewards[0], torch.full((2,), env.reward + expected_bonus)
    )
    metrics = adapter.train_metrics()
    assert metrics["RL/entropy_rate"] == 0.3
    assert metrics["RL/entropy_step_coefficient"] == pytest.approx(0.006)
    assert metrics["Train/entropy_bonus_per_step"] == pytest.approx(expected_bonus)


def test_ppo_counts_episode_endings_without_extending_its_short_rollout():
    adapter, _, env = _ppo_adapter(reward=None, feature_source=None)
    env.done_every = 2
    generation = adapter.collect()
    assert env.step_calls == 3
    assert generation.completed_episodes == 2
    assert generation.episode_mask is None


def test_a_learned_reward_without_a_feature_source_is_refused() -> None:
    with pytest.raises(ValueError, match="feature_source"):
        RslRlPpoAdapter(
            _Ppo(),
            env=_Env(),
            feature_source=None,
            reward=_Reward(),
            steps_per_env_per_cycle=3,
        )


def test_bc_only_training_never_steps_the_environment() -> None:
    adapter, ppo, env = _ppo_adapter(bc_alg=_Bc())

    generation = adapter.collect()
    metrics = adapter.update()

    assert env.step_calls == 0
    assert generation.transition_count == 0
    assert ppo.bc_only_updates == 1 and ppo.updates == 0
    assert metrics["BC/loss"] == 0.75
    assert adapter.eval_metrics() == {"BC/val_loss": 0.125}


def test_an_unrecognized_update_result_fails_instead_of_logging_nan() -> None:
    adapter, ppo, _ = _ppo_adapter()
    ppo.update = lambda: None  # type: ignore[assignment]

    with pytest.raises(TypeError, match="surrogate"):
        adapter.update()


def test_the_actor_facade_does_not_step_the_environment() -> None:
    adapter, ppo, env = _ppo_adapter()
    observations = torch.zeros(2, 4)

    adapter.actor.act(observations)
    adapter.actor.act_inference(observations)

    assert env.step_calls == 0
    assert ppo.actor.stochastic_calls == 1


def test_validation_can_be_paced_by_transitions_instead_of_iterations() -> None:
    """Arms that collect at different rates are only comparable per transition."""
    runner, _, _, logger = _runner(
        validation_interval=0,
        validation_interval_transitions=16,
        validation_steps_per_env=1,
    )

    runner.learn(4)

    # 8 transitions per cycle, so every second cycle crosses the interval.
    validated_at = [
        payload["Summary/transitions"]
        for payload in logger.payloads("validation/summary")
    ]
    assert validated_at == [16.0, 32.0]


def test_a_transition_cadence_does_not_need_an_iteration_interval() -> None:
    """validation_interval=0 used to silently disable validation entirely."""
    runner, _, _, logger = _runner(
        validation_interval=0,
        validation_interval_transitions=8,
        validation_steps_per_env=1,
    )

    runner.learn(1)

    assert logger.payloads("validation/summary"), "a transition cadence must validate"


def test_validation_scores_each_mode_from_the_initial_state_distribution() -> None:
    """A score is only comparable if every run starts where every other did.

    Begun wherever training left the environment, a validation measures an
    arbitrary starting state that drifts over the run.
    """
    runner, _, _, _ = _runner(validation_interval=1, validation_steps_per_env=2)

    runner.learn(1)

    assert runner.env.resets == 2, "one reset per validated policy mode"


def test_the_validation_feature_gap_is_scored_over_episodes_not_the_window() -> None:
    """The expert side is whole demonstrations; the imitator side must match.

    Accumulating a discounted feature return straight through a reset compares
    a quantity the expert's returns never contain.
    """
    runner, _, reward_learner, _ = _runner(
        validation_interval=1, validation_steps_per_env=4
    )
    scored: list[tuple[int, ...]] = []
    original = reward_learner.per_feature_return_mean

    def record(feats, mask):
        scored.append(tuple(feats.shape))
        return original(feats, mask)

    reward_learner.per_feature_return_mean = record
    runner.env.done_every = 2  # an episode ends every other step

    runner.learn(1)

    # Four steps with a reset every second one: episodes of length 2, never 4.
    assert scored, "validation must score the imitator's features"
    assert all(shape[1] == 2 for shape in scored), scored
