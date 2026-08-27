"""Shared collection/replay adapters with distinct SAC and PMD schedules."""

from __future__ import annotations

from typing import Any

import torch

from adapters.rsl_rl_transition_collector import TransitionCollector
from algorithms.off_policy_common import OffPolicyLearner
from algorithms.state_dependent_tanh_actor import TanhPolicyProtocol
from config.off_policy import OffPolicyTrainCfg
from reward_model.fixed import FixedLinearReward
from storage.transition_replay import TransitionReplayBuffer, TransitionRollout


class _OffPolicyAdapter:
    """Common environment collection, replay, warm-up, and checkpoint state."""

    def __init__(
        self,
        *,
        collector: TransitionCollector,
        replay: TransitionReplayBuffer,
        learner: OffPolicyLearner,
        train_cfg: OffPolicyTrainCfg,
        fixed_reward: FixedLinearReward | None,
    ) -> None:
        self.collector = collector
        self.replay = replay
        self.learner = learner
        self.train_cfg = train_cfg
        self.fixed_reward = fixed_reward
        self._warmup_complete = train_cfg.initial_critic_updates == 0
        self._last_metrics: dict[str, float] = {}
        self._pending_rollout: TransitionRollout | None = None
        self._pending_metrics: dict[str, float] | None = None
        self._last_simulator_steps = collector.num_envs * collector.rollout_steps

    @property
    def actor(self) -> TanhPolicyProtocol:
        return self.learner.policy

    @property
    def simulator_steps_per_update(self) -> int:
        return self._last_simulator_steps

    def reset_collection_state(self) -> None:
        self.collector.reset_state(reset_environment=True)

    def _collect(self) -> tuple[TransitionRollout, dict[str, float]]:
        rollout, metrics = self.collector.collect(
            policy_version=self.learner.policy_version
        )
        self._last_simulator_steps = rollout.transition_count
        if self.fixed_reward is not None:
            error = self.fixed_reward.reconstruction_error(
                rollout.features, rollout.environment_rewards
            )
            maximum_error = float(error.max().item())
            if maximum_error > 1.0e-5:
                raise ValueError(
                    "Fixed reward does not reconstruct environment reward: "
                    f"maximum error {maximum_error:.3e}."
                )
            metrics["Reward/reconstruction_max_error"] = maximum_error
        self.replay.add(rollout)
        self.learner.observe_rollout(rollout)
        metrics.update(
            {
                "Data/replay_transitions": float(self.replay.transition_count),
                "Data/simulator_transitions": float(rollout.transition_count),
                "Data/policy_version": float(self.learner.policy_version),
                "Diagnostics/non_finite_count": 0.0,
            }
        )
        oldest_version, newest_version = self.replay.policy_version_range()
        metrics["Data/replay_policy_age_min"] = float(
            self.learner.policy_version - newest_version
        )
        metrics["Data/replay_policy_age_max"] = float(
            self.learner.policy_version - oldest_version
        )
        return rollout, metrics

    def collect_generation(self) -> TransitionRollout:
        """Collect once so IRL may update reward before policy improvement."""
        if self._pending_rollout is not None:
            raise RuntimeError("Previous off-policy generation is still pending.")
        rollout, metrics = self._collect()
        self._pending_rollout = rollout
        self._pending_metrics = metrics
        return rollout

    def _take_cycle(self) -> tuple[TransitionRollout, dict[str, float]]:
        if self._pending_rollout is None:
            return self._collect()
        if self._pending_metrics is None:
            raise RuntimeError("Pending rollout is missing collection metrics.")
        rollout = self._pending_rollout
        metrics = self._pending_metrics
        self._pending_rollout = None
        self._pending_metrics = None
        return rollout, metrics

    def _warm_up_if_needed(self, metrics: dict[str, float]) -> bool:
        if self.replay.transition_count < self.train_cfg.replay.learning_starts:
            metrics["Data/replay_warmup"] = 1.0
            return True
        if self._warmup_complete:
            return False
        actor_before = {
            name: value.detach().clone()
            for name, value in self.learner.policy.model.state_dict().items()
        }
        critic_metrics: dict[str, float] = {}
        for _ in range(self.train_cfg.initial_critic_updates):
            critic_metrics = self.learner.critic_update(self.replay)
        for name, value in self.learner.policy.model.state_dict().items():
            if not torch.equal(value, actor_before[name]):
                raise RuntimeError(
                    f"Actor changed during critic warm-up at parameter {name!r}."
                )
        self._warmup_complete = True
        metrics.update(critic_metrics)
        metrics["Data/initial_critic_updates"] = float(
            self.train_cfg.initial_critic_updates
        )
        return True

    def train_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    def eval_metrics(self) -> dict[str, float]:
        return {}

    def train_mode(self) -> None:
        self.learner.train()

    def eval_mode(self) -> None:
        self.learner.eval()

    def save_state(self) -> dict[str, Any]:
        return {
            "learner": self.learner.save_state(),
            "collector": self.collector.save_state(),
            "warmup_complete": self._warmup_complete,
        }

    def load_state(
        self, checkpoint: dict[str, Any], *, load_optimizer: bool = True
    ) -> None:
        self.learner.load_state(checkpoint["learner"], load_optimizer=load_optimizer)
        self.collector.load_state(checkpoint["collector"])
        self._warmup_complete = bool(checkpoint["warmup_complete"])
        # Replay is deliberately not checkpointed. Collection refills it before
        # any resumed update can sample transitions.
        self.replay.clear()
        self._pending_rollout = None
        self._pending_metrics = None


class SacAdapter(_OffPolicyAdapter):
    """One reparameterized actor step after every critic step."""

    def update(self) -> dict[str, float]:
        rollout, metrics = self._take_cycle()
        if self._warm_up_if_needed(metrics):
            self._last_metrics = metrics
            return dict(metrics)
        update_count = self.train_cfg.critic_updates_for(rollout.transition_count)
        critic_metrics: list[dict[str, float]] = []
        actor_metrics: list[dict[str, float]] = []
        for _ in range(update_count):
            critic_metrics.append(self.learner.critic_update(self.replay))
            actor_metrics.append(self.learner.actor_update(self.replay))
        metrics.update(_mean_metrics(critic_metrics))
        metrics.update(_mean_metrics(actor_metrics))
        metrics["Data/critic_updates"] = float(update_count)
        metrics["Data/actor_updates"] = float(update_count)
        metrics["Data/policy_version"] = float(self.learner.policy_version)
        self._last_metrics = metrics
        return dict(metrics)


class PmdAdapter(_OffPolicyAdapter):
    """Repeated evaluate-then-project cycles over one collected rollout.

    Each cycle spends ``critic_updates_per_projection`` critic steps evaluating
    the frozen pi_k and then replaces it with one bounded projection, so the
    trust-region budget PMD spends per simulator transition is an explicit
    configuration choice instead of one step per rollout.
    """

    def __init__(
        self,
        *,
        collector: TransitionCollector,
        replay: TransitionReplayBuffer,
        learner: OffPolicyLearner,
        train_cfg: OffPolicyTrainCfg,
        fixed_reward: FixedLinearReward | None,
        critic_updates_per_projection: int,
    ) -> None:
        super().__init__(
            collector=collector,
            replay=replay,
            learner=learner,
            train_cfg=train_cfg,
            fixed_reward=fixed_reward,
        )
        if critic_updates_per_projection <= 0:
            raise ValueError("critic_updates_per_projection must be > 0.")
        self.critic_updates_per_projection = int(critic_updates_per_projection)

    def update(self) -> dict[str, float]:
        rollout, metrics = self._take_cycle()
        if self._warm_up_if_needed(metrics):
            self._last_metrics = metrics
            return dict(metrics)
        update_count = self.train_cfg.critic_updates_for(rollout.transition_count)
        projections = max(1, update_count // self.critic_updates_per_projection)
        critic_metrics: list[dict[str, float]] = []
        actor_metrics: list[dict[str, float]] = []
        for _ in range(projections):
            for _ in range(self.critic_updates_per_projection):
                critic_metrics.append(self.learner.critic_update(self.replay))
            actor_metrics.append(self.learner.actor_update(self.replay))
        metrics.update(_mean_metrics(critic_metrics))
        metrics.update(_mean_metrics(actor_metrics))
        metrics["Data/critic_updates"] = float(len(critic_metrics))
        metrics["Data/actor_updates"] = float(projections)
        metrics["Data/policy_version"] = float(self.learner.policy_version)
        self._last_metrics = metrics
        return dict(metrics)


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = set.intersection(*(set(item) for item in items))
    return {key: sum(item[key] for item in items) / len(items) for key in sorted(keys)}
