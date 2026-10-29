from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from conftest import build_rollout  # noqa: E402

from adapters.pmd import PmdAdapter  # noqa: E402
from config.schema import OffPolicyTrainCfg, TransitionReplayCfg  # noqa: E402
from interfaces import FeatureSpec  # noqa: E402
from storage.episode_batch import EpisodeBatch  # noqa: E402
from storage.transition_replay import TransitionReplayBuffer  # noqa: E402


class _CountingLearner:
    """Learner double that only records how often it is asked to update."""

    def __init__(self) -> None:
        self.policy = _FrozenPolicy()
        self.policy_version = 0
        self.critic_updates = 0
        self.actor_updates = 0
        self.observed_rollouts = 0
        self.resets = 0
        self.events: list[str] = []

    def observe_rollout(self, rollout: Any) -> None:
        del rollout
        self.observed_rollouts += 1

    def critic_update(self, replay: Any) -> dict[str, float]:
        del replay
        self.critic_updates += 1
        self.events.append("critic")
        return {"DirectQ/td_loss": 1.0}

    def actor_update(self, replay: Any) -> dict[str, float]:
        del replay
        self.actor_updates += 1
        self.events.append("actor")
        self.policy_version += 1
        return {
            "Actor/loss": 2.0,
            "PMD/projection_steps_taken": 3.0,
            "PMD/collection_policy_kl": float(self.actor_updates),
        }

    def reset_parameters(self) -> None:
        self.resets += 1

    def train(self) -> None: ...

    def eval(self) -> None: ...

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool = True) -> None:
        del state, load_optimizer


class _FrozenPolicy:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(2, 2)


class _StubFeatureTap:
    """Pre-reset tap double exposing only what reward decomposition needs."""

    spec = FeatureSpec(dim=2, names=("alive", "effort"))
    step_dt = 0.5

    @property
    def weights(self) -> torch.Tensor:
        return torch.tensor([1.0, -2.0])


class _StubCollector:
    """Collector double returning identical chunks without an environment."""

    def __init__(self, *, num_envs: int = 4, rollout_steps: int = 8) -> None:
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.feature_tap = _StubFeatureTap()
        self._rollout_id = 0
        self.steps = self
        self.episode_lengths: tuple[int, ...] | None = None
        self.collections = 0

    def collect(self, *, policy_version: int):
        self.collections += 1
        rollout = build_rollout(
            num_envs=self.num_envs,
            horizon=self.rollout_steps,
            rollout_id=self._rollout_id,
            policy_version=policy_version,
            seed=self._rollout_id,
        )
        self._rollout_id += 1
        rollout.terminated.zero_()
        rollout.truncated.zero_()
        lengths = torch.tensor(
            self.episode_lengths or (self.rollout_steps,) * self.num_envs
        )
        rollout.truncated[torch.arange(self.num_envs), lengths - 1] = True
        rollout.bootstrap_valid.fill_(True)
        return EpisodeBatch(
            rollout,
            torch.arange(self.rollout_steps)[None, :] < lengths[:, None],
            rollout.transition_count,
        )

    def maximum_transitions(self):
        return self.num_envs * self.rollout_steps

    def rollout_metrics(self, rollout):
        return {}

    def reset_state(self, *, reset_environment: bool) -> None:
        del reset_environment

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        del state


def _train_cfg(**overrides) -> OffPolicyTrainCfg:
    values = {
        "episodes_per_batch": 4,
        "critic_updates_per_batch": 16,
        "initial_critic_updates": 0,
        "replay": TransitionReplayCfg(learning_starts=1, capacity=100_000),
    }
    values.update(overrides)
    return OffPolicyTrainCfg(**values)


def _cycle(adapter: PmdAdapter) -> dict[str, float]:
    """One outer cycle: collect a frozen-policy batch, then update on it.

    The collection is explicit because the schedule under test is exactly
    "one batch in, one bounded block of updates out"; an update that could
    quietly fetch its own data would make every count below unfalsifiable.
    """
    adapter.collect()
    return adapter.update()


def _adapter(
    train_cfg: OffPolicyTrainCfg, *, policy_improvements_per_batch: int = 1
) -> tuple[_CountingLearner, PmdAdapter]:
    learner = _CountingLearner()
    adapter = PmdAdapter(
        collector=_StubCollector(),
        replay=TransitionReplayBuffer(capacity_transitions=train_cfg.replay.capacity),
        learner=learner,
        train_cfg=train_cfg,
        fixed_reward=None,
        policy_improvements_per_batch=policy_improvements_per_batch,
    )
    return learner, adapter


def test_default_performs_one_improvement_per_episode_batch() -> None:
    train_cfg = _train_cfg()
    learner, adapter = _adapter(train_cfg)

    metrics = _cycle(adapter)

    expected = train_cfg.critic_updates_per_batch
    assert learner.critic_updates == expected
    assert learner.actor_updates == 1
    assert metrics["Data/critic_updates"] == float(expected)
    assert metrics["Data/actor_updates"] == 1.0


def test_training_reports_the_terminal_success_feature(monkeypatch):
    _, adapter = _adapter(_train_cfg())
    batch = adapter.collector.collect(policy_version=0)
    batch.transitions.features.zero_()
    batch.transitions.features[0, -1, 1] = 50.0
    batch.transitions.terminated.zero_()
    batch.transitions.truncated.zero_()
    batch.transitions.truncated[:, -1] = True
    adapter.collector.feature_tap.spec = FeatureSpec(2, ("other", "terminal_success"))
    monkeypatch.setattr(adapter.collector, "collect", lambda **kwargs: batch)
    metrics = _cycle(adapter)
    assert metrics["Train/episode_success_rate"] == 0.25


def test_pmd_spends_exact_independent_budgets_including_remainder() -> None:
    train_cfg = _train_cfg(critic_updates_per_batch=17)
    learner = _CountingLearner()
    adapter = PmdAdapter(
        collector=_StubCollector(),
        replay=TransitionReplayBuffer(capacity_transitions=train_cfg.replay.capacity),
        learner=learner,
        train_cfg=train_cfg,
        fixed_reward=None,
        policy_improvements_per_batch=4,
    )

    metrics = _cycle(adapter)

    critic_updates = train_cfg.critic_updates_per_batch
    projections = 4
    assert projections == 4
    assert learner.actor_updates == projections
    assert learner.critic_updates == critic_updates
    assert metrics["Data/actor_updates"] == float(projections)
    assert metrics["Data/policy_version"] == float(projections)


def test_more_critic_compute_does_not_add_policy_improvements() -> None:
    train_cfg = _train_cfg(critic_updates_per_batch=800)
    learner = _CountingLearner()
    adapter = PmdAdapter(
        collector=_StubCollector(),
        replay=TransitionReplayBuffer(capacity_transitions=train_cfg.replay.capacity),
        learner=learner,
        train_cfg=train_cfg,
        fixed_reward=None,
        policy_improvements_per_batch=1,
    )

    _cycle(adapter)

    assert learner.actor_updates == 1


def test_transition_utd_counts_retained_data_and_interleaves_improvements() -> None:
    learner, adapter = _adapter(
        _train_cfg(critic_updates_per_batch=None, critic_updates_per_transition=2.0),
        policy_improvements_per_batch=4,
    )
    adapter.collector.episode_lengths = (1, 2, 3, 4)

    metrics = _cycle(adapter)

    assert learner.critic_updates == 20  # 10 retained, not 32 physical transitions.
    assert learner.events == (["critic"] * 5 + ["actor"]) * 4
    assert metrics["Data/critic_updates_per_transition"] == 2.0
    assert metrics["Data/critic_updates_per_environment_transition"] == 20 / 32
    assert metrics["Data/actor_optimizer_steps"] == 12
    assert metrics["PMD/collection_policy_kl"] == 4  # Final, not round average.
    adapter.collector.episode_lengths = None
    metrics = _cycle(adapter)
    assert metrics["Data/critic_updates"] == 64


def test_fractional_utd_credit_carries_across_batches_and_checkpoints() -> None:
    cfg = _train_cfg(critic_updates_per_batch=None, critic_updates_per_transition=0.4)
    learner, adapter = _adapter(cfg)
    assert _cycle(adapter)["Data/critic_updates"] == 12
    saved = adapter.save_state()
    assert saved["critic_update_credit"] == pytest.approx(0.8)
    assert _cycle(adapter)["Data/critic_updates"] == 13

    _, restored = _adapter(cfg)
    restored.load_state(saved)
    assert _cycle(restored)["Data/critic_updates"] == 13
    for _ in range(3):
        _cycle(adapter)
    assert learner.critic_updates == 64  # floor(5 * 32 * 0.4)


def test_utd_does_not_charge_replay_warmup_or_count_initial_fit_in_ratio() -> None:
    learner, adapter = _adapter(
        _train_cfg(
            critic_updates_per_batch=None,
            critic_updates_per_transition=2.0,
            initial_critic_updates=7,
            replay=TransitionReplayCfg(learning_starts=64, capacity=1000),
        )
    )
    _cycle(adapter)
    assert learner.critic_updates == 0
    metrics = _cycle(adapter)
    assert learner.critic_updates == 64 + 7
    assert metrics["Data/critic_updates_per_transition"] == 2.0
    assert metrics["Data/initial_critic_updates"] == 7


def test_pmd_rejects_a_non_positive_projection_cadence() -> None:
    train_cfg = _train_cfg()

    with pytest.raises(ValueError, match="policy_improvements_per_batch"):
        PmdAdapter(
            collector=_StubCollector(),
            replay=TransitionReplayBuffer(capacity_transitions=1_000),
            learner=_CountingLearner(),
            train_cfg=train_cfg,
            fixed_reward=None,
            policy_improvements_per_batch=0,
        )


def test_warmup_runs_critic_only_updates_before_any_projection() -> None:
    train_cfg = _train_cfg(initial_critic_updates=7)
    learner = _CountingLearner()
    adapter = PmdAdapter(
        collector=_StubCollector(),
        replay=TransitionReplayBuffer(capacity_transitions=train_cfg.replay.capacity),
        learner=learner,
        train_cfg=train_cfg,
        fixed_reward=None,
        policy_improvements_per_batch=1,
    )

    metrics = _cycle(adapter)

    assert learner.critic_updates == 7 + train_cfg.critic_updates_per_batch
    assert learner.actor_updates == 1
    assert metrics["Data/initial_critic_updates"] == 7.0


def test_demo_warmup_is_before_collection_once_and_does_not_earn_utd_credit() -> None:
    cfg = _train_cfg(initial_critic_updates=7)
    learner, adapter = _adapter(cfg)
    adapter.replay = TransitionReplayBuffer(
        capacity_transitions=1000, demonstration_fraction=0.5
    )
    adapter.replay.seed_with_demonstrations(build_rollout())
    actor_before = {
        name: value.clone() for name, value in learner.policy.model.state_dict().items()
    }
    metrics = adapter.warm_start_from_demonstrations()
    assert metrics["WarmStart/critic_updates"] == 7
    assert learner.critic_updates == 7
    assert learner.actor_updates == 0
    assert adapter.collector.collections == 0
    assert adapter.replay.transition_count == 0
    assert adapter._critic_update_credit == 0
    for name, value in learner.policy.model.state_dict().items():
        torch.testing.assert_close(value, actor_before[name], rtol=0, atol=0)
    assert adapter.warm_start_from_demonstrations() == {}
    assert learner.critic_updates == 7
    online = _cycle(adapter)
    assert online["Data/critic_updates"] == cfg.critic_updates_per_batch
    assert "Data/initial_critic_updates" not in online
    assert learner.critic_updates == 7 + cfg.critic_updates_per_batch
    with pytest.raises(RuntimeError, match="precede online collection"):
        adapter.warm_start_from_demonstrations()


def test_demo_warmup_requires_demonstrations() -> None:
    _, adapter = _adapter(_train_cfg(initial_critic_updates=7))
    with pytest.raises(ValueError, match="seeded demonstration"):
        adapter.warm_start_from_demonstrations()
    assert adapter.collector.collections == 0


def test_an_update_never_collects_its_own_batch() -> None:
    """The reward is fit to the batch, then frozen; an update must not refill it."""
    learner, adapter = _adapter(_train_cfg())

    with pytest.raises(RuntimeError, match="call collect\\(\\) once first"):
        adapter.update()

    assert adapter.collector.collections == 0
    assert adapter.replay.transition_count == 0
    assert learner.critic_updates == 0
    assert learner.actor_updates == 0
    assert learner.observed_rollouts == 0


def test_a_second_update_on_one_batch_spends_no_further_budget() -> None:
    """One collection buys one update block; a repeat is an error, not free UTD."""
    learner, adapter = _adapter(
        _train_cfg(critic_updates_per_batch=None, critic_updates_per_transition=2.0)
    )

    _cycle(adapter)
    spent = (learner.critic_updates, learner.actor_updates)
    replayed = adapter.replay.transition_count

    with pytest.raises(RuntimeError, match="call collect\\(\\) once first"):
        adapter.update()

    assert (learner.critic_updates, learner.actor_updates) == spent
    assert adapter.replay.transition_count == replayed
    assert adapter.collector.collections == 1


def test_pending_generation_is_consumed_by_the_next_update() -> None:
    train_cfg = _train_cfg()
    learner, adapter = _adapter(train_cfg)

    generation = adapter.collect()
    with pytest.raises(RuntimeError, match="still pending"):
        adapter.collect()
    adapter.update()

    assert generation.transition_count == 32
    assert learner.observed_rollouts == 1
    assert adapter.simulator_steps_per_update == 32


def test_learner_is_reset_on_the_configured_interval() -> None:
    train_cfg = _train_cfg(reset_interval_updates=32)
    learner, adapter = _adapter(train_cfg)

    # 16 critic updates per rollout at this ratio: two rollouts per reset.
    first = _cycle(adapter)
    assert learner.resets == 0
    assert first["Data/parameter_resets"] == 0.0

    second = _cycle(adapter)
    assert learner.resets == 1
    assert second["Data/parameter_resets"] == 1.0


def test_resets_are_off_by_default() -> None:
    learner, adapter = _adapter(_train_cfg())

    for _ in range(6):
        metrics = _cycle(adapter)

    assert learner.resets == 0
    assert metrics["Data/parameter_resets"] == 0.0


def test_collection_reports_training_return_and_reward_terms() -> None:
    _, adapter = _adapter(_train_cfg())

    metrics = _cycle(adapter)

    assert "Train/reward_per_step" in metrics
    # step_dt * weight * mean feature, one entry per named term.
    assert set(metrics) >= {"Reward/alive", "Reward/effort"}


def test_critic_scope_reset_leaves_the_actor_alone() -> None:
    """Fine-tuning needs a reset that keeps the behaviour prior it started from."""
    from algorithms.off_policy.actor import (
        StateDependentTanhGaussianActor,
        StateDependentTanhPolicy,
    )
    from algorithms.off_policy.pmd import Pmd
    from config.schema import PmdCfg

    class _StubValue:
        num_heads = 1

        def __init__(self) -> None:
            self.resets = 0

        def reset_parameters(self) -> None:
            self.resets += 1

    def build(scope: str) -> tuple[Pmd, _StubValue, torch.Tensor]:
        torch.manual_seed(0)
        model = StateDependentTanhGaussianActor(3, 2, hidden_dims=(8,))
        value = _StubValue()
        learner = Pmd(
            StateDependentTanhPolicy(model),
            value,
            cfg=PmdCfg(train=_train_cfg(reset_scope=scope)),
            step_dt=1.0,
            device=torch.device("cpu"),
        )
        return learner, value, model.mean_head.weight.detach().clone()

    critic_learner, critic_value, actor_before = build("critic")
    critic_learner.reset_parameters()
    torch.testing.assert_close(
        critic_learner.policy.model.mean_head.weight, actor_before
    )
    assert critic_value.resets == 1

    agent_learner, agent_value, agent_before = build("agent")
    agent_learner.reset_parameters()
    assert not torch.allclose(agent_learner.policy.model.mean_head.weight, agent_before)
    assert agent_value.resets == 1
