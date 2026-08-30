from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from conftest import build_rollout  # noqa: E402

from adapters.off_policy import PmdAdapter, SacAdapter  # noqa: E402
from config.off_policy import OffPolicyTrainCfg, TransitionReplayCfg  # noqa: E402
from interfaces import FeatureSpec, PhasedOffPolicyAlgorithm  # noqa: E402
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

    def observe_rollout(self, rollout: Any) -> None:
        del rollout
        self.observed_rollouts += 1

    def critic_update(self, replay: Any) -> dict[str, float]:
        del replay
        self.critic_updates += 1
        return {"DirectQ/td_loss": 1.0}

    def actor_update(self, replay: Any) -> dict[str, float]:
        del replay
        self.actor_updates += 1
        self.policy_version += 1
        return {"Actor/loss": 2.0}

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

    def collect(self, *, policy_version: int):
        rollout = build_rollout(
            num_envs=self.num_envs,
            horizon=self.rollout_steps,
            rollout_id=self._rollout_id,
            policy_version=policy_version,
            seed=self._rollout_id,
        )
        self._rollout_id += 1
        return rollout, {}

    def reset_state(self, *, reset_environment: bool) -> None:
        del reset_environment

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        del state


def _train_cfg(**overrides) -> OffPolicyTrainCfg:
    values = {
        "rollout_steps": 8,
        "updates_per_transition": 0.5,
        "initial_critic_updates": 0,
        "replay": TransitionReplayCfg(learning_starts=1, capacity=100_000),
    }
    values.update(overrides)
    return OffPolicyTrainCfg(**values)


def _sac_adapter(train_cfg: OffPolicyTrainCfg) -> tuple[_CountingLearner, SacAdapter]:
    learner = _CountingLearner()
    adapter = SacAdapter(
        collector=_StubCollector(),
        replay=TransitionReplayBuffer(capacity_transitions=train_cfg.replay.capacity),
        learner=learner,
        train_cfg=train_cfg,
        fixed_reward=None,
    )
    return learner, adapter


def test_sac_pairs_one_actor_step_with_every_critic_step() -> None:
    train_cfg = _train_cfg()
    learner, adapter = _sac_adapter(train_cfg)

    metrics = adapter.update()

    expected = train_cfg.critic_updates_for(4 * 8)
    assert learner.critic_updates == expected
    assert learner.actor_updates == expected
    assert metrics["Data/critic_updates"] == float(expected)
    assert metrics["Data/actor_updates"] == float(expected)


def test_pmd_projects_once_per_configured_block_of_critic_steps() -> None:
    train_cfg = _train_cfg()
    learner = _CountingLearner()
    adapter = PmdAdapter(
        collector=_StubCollector(),
        replay=TransitionReplayBuffer(capacity_transitions=train_cfg.replay.capacity),
        learner=learner,
        train_cfg=train_cfg,
        fixed_reward=None,
        critic_updates_per_projection=4,
    )

    metrics = adapter.update()

    critic_updates = train_cfg.critic_updates_for(4 * 8)
    projections = critic_updates // 4
    assert projections == 4
    assert learner.actor_updates == projections
    assert learner.critic_updates == projections * 4
    assert metrics["Data/actor_updates"] == float(projections)
    assert metrics["Data/policy_version"] == float(projections)


def test_pmd_always_projects_at_least_once() -> None:
    train_cfg = _train_cfg(updates_per_transition=1.0 / 32.0)
    learner = _CountingLearner()
    adapter = PmdAdapter(
        collector=_StubCollector(),
        replay=TransitionReplayBuffer(capacity_transitions=train_cfg.replay.capacity),
        learner=learner,
        train_cfg=train_cfg,
        fixed_reward=None,
        critic_updates_per_projection=8,
    )

    adapter.update()

    assert learner.actor_updates == 1


def test_pmd_rejects_a_non_positive_projection_cadence() -> None:
    train_cfg = _train_cfg()

    with pytest.raises(ValueError, match="critic_updates_per_projection"):
        PmdAdapter(
            collector=_StubCollector(),
            replay=TransitionReplayBuffer(capacity_transitions=1_000),
            learner=_CountingLearner(),
            train_cfg=train_cfg,
            fixed_reward=None,
            critic_updates_per_projection=0,
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
        critic_updates_per_projection=4,
    )

    metrics = adapter.update()

    assert learner.critic_updates == 7
    assert learner.actor_updates == 0
    assert metrics["Data/initial_critic_updates"] == 7.0


def test_pending_generation_is_consumed_by_the_next_update() -> None:
    train_cfg = _train_cfg()
    learner, adapter = _sac_adapter(train_cfg)

    generation = adapter.collect_generation()
    with pytest.raises(RuntimeError, match="still pending"):
        adapter.collect_generation()
    adapter.update()

    assert generation.transition_count == 32
    assert learner.observed_rollouts == 1
    assert adapter.simulator_steps_per_update == 32


def test_off_policy_adapters_select_the_phased_runner_path() -> None:
    train_cfg = _train_cfg()
    _, adapter = _sac_adapter(train_cfg)

    class _OnPolicyAdapter:
        def update(self) -> dict[str, float]:
            return {}

    assert isinstance(adapter, PhasedOffPolicyAlgorithm)
    assert not isinstance(_OnPolicyAdapter(), PhasedOffPolicyAlgorithm)


def test_learner_is_reset_on_the_configured_interval() -> None:
    train_cfg = _train_cfg(reset_interval_updates=32)
    learner, adapter = _sac_adapter(train_cfg)

    # 16 critic updates per rollout at this ratio: two rollouts per reset.
    first = adapter.update()
    assert learner.resets == 0
    assert first["Data/parameter_resets"] == 0.0

    second = adapter.update()
    assert learner.resets == 1
    assert second["Data/parameter_resets"] == 1.0


def test_resets_are_off_by_default() -> None:
    learner, adapter = _sac_adapter(_train_cfg())

    for _ in range(6):
        metrics = adapter.update()

    assert learner.resets == 0
    assert metrics["Data/parameter_resets"] == 0.0


def test_collection_reports_training_return_and_reward_terms() -> None:
    _, adapter = _sac_adapter(_train_cfg())

    metrics = adapter.update()

    assert "Train/reward_per_step" in metrics
    # step_dt * weight * mean feature, one entry per named term.
    assert set(metrics) >= {"Reward/alive", "Reward/effort"}


def test_critic_scope_reset_leaves_the_actor_alone() -> None:
    """Fine-tuning needs a reset that keeps the behaviour prior it started from."""
    from algorithms.sac import Sac
    from algorithms.state_dependent_tanh_actor import (
        StateDependentTanhGaussianActor,
        StateDependentTanhPolicy,
    )
    from config.off_policy import SacCfg

    class _StubValue:
        num_heads = 1

        def __init__(self) -> None:
            self.resets = 0

        def reset_parameters(self) -> None:
            self.resets += 1

    def build(scope: str) -> tuple[Sac, _StubValue, torch.Tensor]:
        torch.manual_seed(0)
        model = StateDependentTanhGaussianActor(3, 2, hidden_dims=(8,))
        value = _StubValue()
        learner = Sac(
            StateDependentTanhPolicy(model),
            value,
            cfg=SacCfg(train=_train_cfg(reset_scope=scope)),
            action_dim=2,
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
