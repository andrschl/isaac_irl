"""Environment collection, replay, and the PMD update schedule."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch

from adapters.episode_collector import EpisodeCollector
from algorithms.off_policy.actor import TanhPolicyProtocol
from algorithms.off_policy.common import OffPolicyLearner
from config.schema import OffPolicyTrainCfg
from interfaces import Generation
from reward.episode_success import episode_success
from reward.fixed import FixedLinearReward
from storage.episode_batch import EpisodeBatch
from storage.transition_replay import TransitionReplayBuffer, TransitionRollout


class PmdAdapter:
    """Collect, replay, and run repeated evaluate-then-project PMD cycles.

    Collect complete episodes with a frozen actor. Critic compute and bounded
    policy improvements are independent per-batch budgets.
    """

    def __init__(
        self,
        *,
        collector: EpisodeCollector,
        replay: TransitionReplayBuffer,
        learner: OffPolicyLearner,
        train_cfg: OffPolicyTrainCfg,
        fixed_reward: FixedLinearReward | None,
        policy_improvements_per_batch: int,
    ) -> None:
        if not (
            0
            < policy_improvements_per_batch
            <= train_cfg.minimum_critic_updates_per_batch
        ):
            raise ValueError(
                "policy_improvements_per_batch must be positive and no greater "
                "than the minimum critic budget per batch."
            )
        self.policy_improvements_per_batch = int(policy_improvements_per_batch)
        self.collector = collector
        self.replay = replay
        self.learner = learner
        self.train_cfg = train_cfg
        self.fixed_reward = fixed_reward
        self._warmup_complete = train_cfg.initial_critic_updates == 0
        self._last_metrics: dict[str, float] = {}
        self._pending_rollout: TransitionRollout | None = None
        self._pending_metrics: dict[str, float] | None = None
        self._pending_episodes: EpisodeBatch | None = None
        self._critic_updates_since_reset = 0
        self._critic_update_credit = 0.0
        self._resets = 0

    @property
    def actor(self) -> TanhPolicyProtocol:
        return self.learner.policy

    @property
    def simulator_steps_per_update(self) -> int:
        return self.collector.maximum_transitions()

    def reset_collection_state(self) -> None:
        self.collector.steps.invalidate_observations()

    def _collect(self) -> tuple[TransitionRollout, dict[str, float]]:
        episodes = self.collector.collect(policy_version=self.learner.policy_version)
        self._pending_episodes = episodes
        rollout = episodes.valid_rollout()
        metrics = self.collector.steps.rollout_metrics(rollout)
        metrics.pop("Data/rollout_steps_per_env", None)
        metrics.update(
            {
                "Data/completed_episodes": float(episodes.episode_count),
                "Data/episode_length_mean": float(episodes.mask.sum(1).float().mean()),
                "Data/discarded_transitions": float(
                    episodes.environment_transitions - episodes.transition_count
                ),
                "Time/collection_seconds": episodes.collection_seconds,
                "Time/reset_seconds": episodes.reset_seconds,
                "Time/aggregate_interaction_seconds": episodes.environment_transitions
                * self.collector.steps.feature_tap.step_dt,
                "Train/episode_return_mean": float(episodes.returns.mean()),
            }
        )
        names = self.collector.steps.feature_tap.spec.names or ()
        success = episode_success(
            episodes.transitions.features,
            names,
            episodes.mask,
            terminated=episodes.transitions.terminated,
            truncated=episodes.transitions.truncated,
        )
        if success is not None:
            metrics["Train/episode_success_rate"] = float(success.float().mean())
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
                "Data/simulator_transitions": float(episodes.environment_transitions),
                "Data/policy_version": float(self.learner.policy_version),
                "Data/collection_policy_version": float(rollout.policy_version),
                # What the behaviour policy actually collects, as opposed to
                # what the deterministic evaluation policy scores later.
                "Train/reward_per_step": float(
                    rollout.environment_rewards.mean().item()
                ),
            }
        )
        metrics.update(self._reward_term_metrics(rollout))
        oldest_version, newest_version = self.replay.policy_version_range()
        metrics["Data/replay_policy_age_min"] = float(
            self.learner.policy_version - newest_version
        )
        metrics["Data/replay_policy_age_max"] = float(
            self.learner.policy_version - oldest_version
        )
        return rollout, metrics

    def _reward_term_metrics(self, rollout: TransitionRollout) -> dict[str, float]:
        """Split the environment reward into its named terms, per step.

        Shapes: features [N,T,D] reduced to one mean contribution per term. A
        return that turns negative says nothing about why; the term that did it
        does.
        """
        feature_tap = self.collector.steps.feature_tap
        names = feature_tap.spec.names
        if names is None:
            return {}
        weights = feature_tap.weights.to(rollout.features)
        contributions = (
            feature_tap.step_dt * rollout.features.mean(dim=(0, 1)) * weights
        )
        return {
            f"Reward/{name}": float(value)
            for name, value in zip(names, contributions.tolist(), strict=True)
        }

    def collect(self) -> Generation:
        """Collect once so the reward learner updates before the policy does."""
        if self._pending_rollout is not None:
            raise RuntimeError("Previous off-policy generation is still pending.")
        rollout, metrics = self._collect()
        self._pending_rollout = rollout
        self._pending_metrics = metrics
        episodes = self._pending_episodes
        if episodes is None:
            raise RuntimeError("Episode collection returned no batch.")
        return Generation(
            transition_count=rollout.transition_count,
            features=episodes.transitions.features,
            # Either kind of ending closes an episode for a feature return,
            # even though only a termination stops the Bellman backup.
            dones=episodes.transitions.terminated | episodes.transitions.truncated,
            episode_mask=episodes.mask,
            environment_transition_count=episodes.environment_transitions,
            completed_episodes=episodes.episode_count,
            policy_version=rollout.policy_version,
        )

    def _take_cycle(self) -> tuple[TransitionRollout, dict[str, float]]:
        if self._pending_rollout is None:
            raise RuntimeError(
                "PMD update requires a pending batch; call collect() once first."
            )
        if self._pending_metrics is None:
            raise RuntimeError("Pending rollout is missing collection metrics.")
        rollout = self._pending_rollout
        metrics = self._pending_metrics
        self._pending_rollout = None
        self._pending_metrics = None
        self._pending_episodes = None
        return rollout, metrics

    def _warm_up_if_needed(self, metrics: dict[str, float]) -> bool:
        if self.replay.transition_count < self.train_cfg.replay.learning_starts:
            metrics["Data/replay_warmup"] = 1.0
            return True
        if self._warmup_complete:
            return False
        self._fit_initial_critics(metrics)
        return False

    def warm_start_from_demonstrations(self) -> dict[str, float]:
        """Fit both value critics against the frozen BC policy before interaction.

        The replay must contain demonstrations and no online transitions. The
        normal mixed sampler resumes automatically after the first collection.
        This consumes neither collection nor regular UTD credit.
        """
        if self.replay.demonstration_count == 0:
            raise ValueError("Demo warm-start requires seeded demonstration replay.")
        if self.replay.transition_count or self._pending_rollout is not None:
            raise RuntimeError("Demo warm-start must precede online collection.")
        if self._warmup_complete:
            return {}
        metrics: dict[str, float] = {}
        self._fit_initial_critics(metrics, report_progress=True)
        self._critic_updates_since_reset += self.train_cfg.initial_critic_updates
        return {
            "WarmStart/demonstration_transitions": float(
                self.replay.demonstration_count
            ),
            "WarmStart/critic_updates": float(self.train_cfg.initial_critic_updates),
            "WarmStart/online_transitions": 0.0,
            **{f"WarmStart/{name}": value for name, value in metrics.items()},
        }

    def _fit_initial_critics(
        self, metrics: dict[str, float], *, report_progress: bool = False
    ) -> None:
        """Spend the one-time critic budget without changing policy parameters."""
        actor_before = {
            name: value.detach().clone()
            for name, value in self.learner.policy.model.state_dict().items()
        }
        critic_metrics: dict[str, float] = {}
        for step in range(1, self.train_cfg.initial_critic_updates + 1):
            critic_metrics = self.learner.critic_update(self.replay)
            if step == 1:
                metrics.update(
                    {f"Initial/{name}": value for name, value in critic_metrics.items()}
                )
            if report_progress and (
                step == 1
                or step % 250 == 0
                or step == self.train_cfg.initial_critic_updates
            ):
                losses = {
                    name: round(value, 6)
                    for name, value in critic_metrics.items()
                    if name.endswith("td_loss")
                }
                print(
                    f"[INFO] demo-only critic warm-up {step}/{self.train_cfg.initial_critic_updates}: {losses}",
                    flush=True,
                )
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

    def _reset_if_due(self, critic_updates: int) -> dict[str, float]:
        """Reinitialize the learner once its reset interval has elapsed."""
        interval = self.train_cfg.reset_interval_updates
        self._critic_updates_since_reset += critic_updates
        if interval > 0 and self._critic_updates_since_reset >= interval:
            self.learner.reset_parameters()
            self._critic_updates_since_reset = 0
            self._resets += 1
        return {"Data/parameter_resets": float(self._resets)}

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
            "critic_update_credit": self._critic_update_credit,
        }

    def load_state(
        self, checkpoint: dict[str, Any], *, load_optimizer: bool = True
    ) -> None:
        self.learner.load_state(checkpoint["learner"], load_optimizer=load_optimizer)
        self.collector.load_state(checkpoint["collector"])
        self._warmup_complete = bool(checkpoint["warmup_complete"])
        self._critic_update_credit = float(checkpoint.get("critic_update_credit", 0.0))
        # Replay is deliberately not checkpointed. Collection refills it before
        # any resumed update can sample transitions.
        self.replay.clear()
        self._pending_rollout = None
        self._pending_metrics = None
        self._pending_episodes = None

    def evaluate_episodes(self, mode_name: str, episodes: int) -> EpisodeBatch:
        """Collect an evaluation batch only after training consumed its batch."""
        if self._pending_rollout is not None:
            raise RuntimeError("Finish the pending training update before evaluation.")
        actions = {"inference": self.actor.act_inference, "stochastic": self.actor.act}
        if mode_name not in actions:
            raise ValueError(f"Unknown evaluation mode {mode_name!r}.")
        device = self.collector.steps.device
        # Preserve training RNG streams, not simulator state. The next training
        # batch explicitly resets; arbitrary env-private RNGs are not captured.
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        try:
            with torch.random.fork_rng(
                devices=[device] if device.type == "cuda" else []
            ):
                return self.collector.collect(
                    policy_version=self.learner.policy_version,
                    episodes=episodes,
                    action_fn=actions[mode_name],
                )
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)

    def _critic_update_budget(self, new_transitions: int) -> int:
        """Count optimizer steps; retain fractional UTD credit across batches."""
        fixed_budget = self.train_cfg.critic_updates_per_batch
        if fixed_budget is not None:
            return fixed_budget
        budget = (
            self.train_cfg.critic_updates_per_transition * new_transitions
            + self._critic_update_credit
        )
        updates = math.floor(budget)
        self._critic_update_credit = budget - updates
        return updates

    def update(self) -> dict[str, float]:
        rollout, metrics = self._take_cycle()
        if self._warm_up_if_needed(metrics):
            self._last_metrics = metrics
            return dict(metrics)
        update_count = self._critic_update_budget(rollout.transition_count)
        projections = self.policy_improvements_per_batch
        if update_count < projections:
            raise ValueError(
                f"Collected {rollout.transition_count} transitions, giving only "
                f"{update_count} critic steps for {projections} policy improvements."
            )
        steps_per_projection, remainder = divmod(update_count, projections)
        critic_metrics: list[dict[str, float]] = []
        actor_metrics: list[dict[str, float]] = []
        for projection in range(projections):
            for _ in range(steps_per_projection + (projection < remainder)):
                critic_metrics.append(self.learner.critic_update(self.replay))
            actor_metrics.append(self.learner.actor_update(self.replay))
        metrics.update(_mean_metrics(critic_metrics))
        metrics.update(_mean_metrics(actor_metrics))
        # Drift is relative to the collection policy, not the preceding round.
        # Keep the final cumulative value instead of averaging over rounds.
        metrics.update(
            {
                name: value
                for name, value in actor_metrics[-1].items()
                if name.startswith("PMD/collection_policy_")
            }
        )
        metrics["Data/critic_updates"] = float(len(critic_metrics))
        metrics["Data/critic_updates_per_transition"] = (
            len(critic_metrics) / rollout.transition_count
        )
        metrics["Data/critic_updates_per_environment_transition"] = (
            len(critic_metrics) / metrics["Data/simulator_transitions"]
        )
        metrics["Data/actor_optimizer_steps"] = sum(
            item.get("PMD/projection_steps_taken", 0.0) for item in actor_metrics
        )
        metrics["Data/actor_updates"] = float(projections)
        metrics["Data/policy_version"] = float(self.learner.policy_version)
        metrics.update(
            self._reset_if_due(
                len(critic_metrics) + int(metrics.get("Data/initial_critic_updates", 0))
            )
        )
        self._last_metrics = metrics
        return dict(metrics)


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = set.intersection(*(set(item) for item in items))
    return {key: sum(item[key] for item in items) / len(items) for key in sorted(keys)}
