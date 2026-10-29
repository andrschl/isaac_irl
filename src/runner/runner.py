"""The single training runner: collect, learn a reward, improve the policy."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Callable

import torch

from algorithms.irl import IrlAlgorithm
from config.schema import RunnerCfg
from interfaces import FeatureSource, Generation, RlAlgorithm, RuntimeContext
from reward.episode_success import episode_success
from reward.features import FeatureColumns
from runner.loggers import LogEvent, Logger, NoopLogger
from runner.summary import (
    TRANSITIONS,
    SampleEfficiencyTracker,
    training_summary,
)
from storage.episode_batch import EpisodeBatch
from storage.feature_storage import FeatureTrajectoryBuffer
from storage.observations import Observation, detached_observation

ActionFn = Callable[[Observation], torch.Tensor]


@dataclass(frozen=True, slots=True)
class _Rollout:
    """What one evaluation rollout produced.

    Shapes: features [N,T,D], dones [N,T], total_reward [N], ever_success [N].
    """

    features: torch.Tensor
    dones: torch.Tensor
    total_reward: torch.Tensor
    ever_success: torch.Tensor | None
    episode_mask: torch.Tensor | None = None


SuccessFn = Callable[[Any], torch.Tensor]


class Runner:
    """Drive one learning run: the policy, the reward learner, and reporting.

    One iteration is one cycle: the algorithm collects a generation of
    environment transitions, the reward learner updates on them (when a reward
    is being learned), and the algorithm then improves its policy. Validation
    scores both policy modes on the environment's own reward, so a run is
    comparable across reward sources and algorithms.
    """

    def __init__(
        self,
        env: Any,
        *,
        rl_alg: RlAlgorithm,
        reward_learner: IrlAlgorithm | None,
        feature_source: FeatureSource,
        cfg: RunnerCfg,
        success_fn: SuccessFn | None = None,
        log_dir: str | None = None,
        device: str | torch.device = "cpu",
        logger: Logger | None = None,
        episode_evaluator: Callable[[str, int], EpisodeBatch] | None = None,
        reward_features: FeatureColumns | None = None,
    ) -> None:
        self.env = env
        self.rl_alg = rl_alg
        self.reward_learner = reward_learner
        self.feature_source = feature_source
        self.reward_features = reward_features
        self.cfg = cfg
        self.episode_evaluator = episode_evaluator
        self.success_fn = success_fn
        self.log_dir = log_dir
        self.device = torch.device(device)
        self.logger: Logger = logger or NoopLogger()

        self.current_iteration = 0
        self.global_timestep = 0
        self.environment_transitions = 0
        self.evaluation_transitions = 0
        self.imitator_transitions = 0
        self.training_episodes = 0
        self.evaluation_episodes = 0
        self._last_metrics: dict[str, float] = {}
        # Set from learn(init_at_random_ep_len=...); off until a run starts.
        self._stagger_resets = False
        self._next_validation_transitions = int(
            cfg.validation_interval_transitions or 0
        )
        self._runtime_ctx = RuntimeContext(
            num_envs=int(env.num_envs),
            feature_dim=int(feature_source.spec.dim),
            device=str(self.device),
        )
        self._sample_efficiency = SampleEfficiencyTracker(
            validation_steps=max(1, self._validation_steps()),
            return_per_step_thresholds=cfg.return_per_step_thresholds,
            success_rate_thresholds=cfg.success_rate_thresholds,
        )

    # ---------------------------------------------------------------- cycle

    def learn(
        self, num_iterations: int, *, init_at_random_ep_len: bool = False
    ) -> None:
        """Run cycles until the iteration count or transition budget is met."""
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be > 0, got {num_iterations}.")
        self.logger.start(num_iterations)
        self._stagger_resets = (
            bool(init_at_random_ep_len) and self.episode_evaluator is None
        )
        # Reset up front rather than letting the first collect() do it lazily:
        # a stagger applied before that reset would be wiped by it.
        self._restart_collection()
        self.rl_alg.train_mode()

        budget = self.cfg.transition_budget
        final_iteration = self.current_iteration + num_iterations
        while self.current_iteration < final_iteration:
            reserve = self.rl_alg.simulator_steps_per_update
            if (
                self.reward_learner is not None
                and self.cfg.imitator_source == "rollout"
            ):
                reserve += self._imitator_rollout_steps() * self._runtime_ctx.num_envs
            if not self._fits_interaction_budget(reserve):
                print(
                    "[INFO] Stopping before the next complete collection batch would exceed total_transition_budget.",
                    flush=True,
                )
                break
            self._run_cycle()
            self.current_iteration += 1
            self._log_iteration()
            if self._should_validate():
                self._validate()
            self._maybe_save_checkpoint()
            if budget is not None and self.global_timestep >= budget:
                break

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_iteration}.pt"))

    def _run_cycle(self) -> None:
        """Collect once, update the reward, then improve the policy."""
        generation = self.rl_alg.collect()
        self.global_timestep += int(generation.transition_count)
        self.environment_transitions += int(
            generation.transition_count
            if generation.environment_transition_count is None
            else generation.environment_transition_count
        )
        self.training_episodes += generation.completed_episodes
        metrics: dict[str, float] = {
            "Data/collected_transitions": float(generation.transition_count)
        }
        if self.reward_learner is not None:
            self._refresh_imitator(generation)
            metrics.update(self._run_reward_updates(generation))
        for _ in range(self.cfg.rl_updates_per_cycle):
            metrics.update(self.rl_alg.update())
        metrics.update(self.rl_alg.train_metrics())
        if self.reward_learner is not None:
            metrics.update(self.reward_learner.train_metrics())
        self._last_metrics = metrics

    def _refresh_imitator(self, generation: Generation) -> None:
        """Give the reward learner the current policy's feature trajectories."""
        assert self.reward_learner is not None
        if generation.episode_mask is not None:
            if generation.features is None:
                raise RuntimeError("Episode masks require collected features.")
            self.reward_learner.replace_imitator_episodes(
                self._reward_feature_columns(generation.features),
                generation.episode_mask,
            )
            return
        if self.cfg.imitator_source == "rollout":
            self._collect_imitator_rollout()
            return
        if generation.features is None or generation.dones is None:
            raise RuntimeError(
                "runner.imitator_source='collected' needs the algorithm to return "
                "features and dones from collect(); it returned none. Use "
                "'rollout' instead."
            )
        self.reward_learner.replace_imitator_trajectories(
            self._reward_feature_columns(generation.features), generation.dones
        )

    def _collect_imitator_rollout(self) -> None:
        """Roll the policy out for whole episodes to refresh imitator features."""
        assert self.reward_learner is not None
        rollout = self._rollout(
            self.rl_alg.actor.act,
            num_steps_per_env=self._imitator_rollout_steps(),
            purpose="imitator",
        )
        self.reward_learner.replace_imitator_trajectories(
            self._reward_feature_columns(rollout.features), rollout.dones
        )
        self._restart_collection()

    def _reward_feature_columns(self, features: torch.Tensor) -> torch.Tensor:
        """Narrow collected features [...,D] to the reward learner's basis."""
        if self.reward_features is None:
            return features
        return self.reward_features.apply(features)

    @property
    def _reward_feature_dim(self) -> int:
        """Width of the reward learner's feature basis, after column selection."""
        if self.reward_features is None:
            return self._runtime_ctx.feature_dim
        return self.reward_features.dim

    def _run_reward_updates(self, generation: Generation) -> dict[str, float]:
        """Fit the reward to this generation, and say what the fit cost.

        The optimizer-step count is summed rather than overwritten: with
        ``reward_updates_per_cycle`` above one, the last update's own count is
        not what the cycle spent. The feature expectations being matched belong
        to the policy that *collected* these episodes, not to the policy the
        following PMD block leaves behind, so that version is logged next to
        them - more reward steps fit a fixed measured gap more exactly, they do
        not make the measured gap smaller.
        """
        assert self.reward_learner is not None
        metrics: dict[str, float] = {}
        if generation.policy_version is not None:
            metrics["IRL/collection_policy_version"] = float(generation.policy_version)
        if not self.reward_learner.can_update():
            return metrics
        optimizer_steps = 0.0
        exact_solves = 0.0
        for _ in range(self.cfg.reward_updates_per_cycle):
            update_metrics = self.reward_learner.update()
            optimizer_steps += float(
                update_metrics.pop("IRL/reward_optimizer_steps", 0.0)
            )
            exact_solves += float(update_metrics.pop("IRL/reward_exact_solves", 0.0))
            metrics.update(update_metrics)
        metrics["IRL/reward_optimizer_steps"] = optimizer_steps
        if exact_solves:
            metrics["IRL/reward_exact_solves"] = exact_solves
        metrics["IRL/reward_updates"] = float(self.cfg.reward_updates_per_cycle)
        return metrics

    def _imitator_rollout_steps(self) -> int:
        if self.cfg.imitator_rollout_steps_per_env is not None:
            return int(self.cfg.imitator_rollout_steps_per_env)
        max_episode_length = int(self.env.max_episode_length)
        if max_episode_length <= 0:
            raise ValueError(
                "env.max_episode_length must be > 0 to size the imitator rollout, "
                f"got {max_episode_length}. Set runner.imitator_rollout_steps_per_env."
            )
        return max_episode_length

    def _restart_collection(self) -> None:
        """Invalidate cached observations; only PPO uses randomized initial ages.

        Episode collectors own explicit resets at wave boundaries. This method
        must never reset their environment or alter their episode ages.
        """
        self.rl_alg.reset_collection_state()
        self._stagger_episode_lengths()

    def _stagger_episode_lengths(self) -> None:
        """Start environments at random episode ages so resets desynchronize.

        Environments that reset in lockstep make every reward that arrives just
        after a reset arrive in all of them at once, so the replay alternates
        between blocks that contain it and blocks that do not. On Lift that is
        most of the reward: the cube is still falling for the first steps of an
        episode, which fires the lifting and goal-tracking terms for free.
        Staggered starts spread that across the buffer instead.

        The ages are randomized after the reset, not before: the environments
        are all physically in their reset pose here, but they now time out at
        different steps, so from the first episode onward they come apart.
        """
        if not self._stagger_resets:
            return
        if not hasattr(self.env, "episode_length_buf"):
            return
        self.env.episode_length_buf = torch.randint_like(
            self.env.episode_length_buf, high=int(self.env.max_episode_length)
        )

    # ------------------------------------------------------------ rollouts

    def _observations(self) -> Observation:
        return detached_observation(self.env.get_observations(), self.device)

    def _rollout(
        self,
        action_fn: ActionFn,
        *,
        num_steps_per_env: int,
        from_reset: bool = False,
        purpose: str = "evaluation",
    ) -> "_Rollout":
        """Roll one policy out without training.

        ``from_reset`` starts from the environment's initial-state
        distribution, which is what makes a score comparable across runs: a
        rollout begun wherever training happened to leave the environment
        measures a different, arbitrary starting state each time.
        """
        num_envs = self._runtime_ctx.num_envs
        if from_reset:
            self.env.reset()
        observations = self._observations()
        step_features: list[torch.Tensor] = []
        step_dones: list[torch.Tensor] = []
        total_reward = torch.zeros(num_envs, device=self.device)
        ever_success = (
            None
            if self.success_fn is None
            else torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        )
        with torch.no_grad():
            for _ in range(num_steps_per_env):
                actions = action_fn(observations)
                next_observations, rewards, dones, _ = self.env.step(
                    actions.to(self.env.device)
                )
                self.environment_transitions += num_envs
                if purpose == "imitator":
                    self.imitator_transitions += num_envs
                else:
                    self.evaluation_transitions += num_envs
                observations = detached_observation(next_observations, self.device)
                total_reward += torch.as_tensor(rewards, device=self.device).reshape(-1)
                step_features.append(self._step_features())
                step_dones.append(
                    torch.as_tensor(dones, device=self.device)
                    .reshape(-1)
                    .to(torch.bool)
                )
                if ever_success is not None:
                    ever_success |= self._step_success()
        return _Rollout(
            features=torch.stack(step_features, dim=1),
            dones=torch.stack(step_dones, dim=1),
            total_reward=total_reward,
            ever_success=ever_success,
        )

    def _step_features(self) -> torch.Tensor:
        features = self.feature_source.features(self.env).to(self.device)
        expected = (self._runtime_ctx.num_envs, self._runtime_ctx.feature_dim)
        if tuple(features.shape) != expected:
            raise ValueError(
                f"Expected feature_source.features(env) {expected}, got "
                f"{tuple(features.shape)}."
            )
        return features

    def _step_success(self) -> torch.Tensor:
        assert self.success_fn is not None
        return (
            torch.as_tensor(self.success_fn(self.env), device=self.device)
            .reshape(-1)
            .to(torch.bool)
        )

    def record_final_rollouts(
        self,
        *,
        recorder: Any,
        video_length: int,
        mode_names: tuple[str, ...] = ("inference", "stochastic"),
    ) -> None:
        """Drive the ``RecordVideo``-wrapped env for one clip per policy mode."""
        if video_length <= 0:
            return
        reserve = len(mode_names) * video_length * self._runtime_ctx.num_envs
        if not self._fits_interaction_budget(reserve):
            self._emit("video/skipped", {"Data/video_skipped_budget": 1.0})
            return
        self.rl_alg.eval_mode()
        try:
            for mode_name in mode_names:
                self.rl_alg.reset_collection_state()
                recorder.start_recording(f"final_{mode_name}")
                try:
                    self._rollout(
                        self._action_fn(mode_name), num_steps_per_env=video_length
                    )
                finally:
                    if recorder.recording:
                        recorder.stop_recording()
        finally:
            self.rl_alg.train_mode()
            self._restart_collection()
        self._emit("video/summary", {})
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_iteration}.pt"))

    def _action_fn(self, mode_name: str) -> ActionFn:
        actor = self.rl_alg.actor
        if mode_name == "inference":
            return actor.act_inference
        if mode_name == "stochastic":
            return actor.act
        raise ValueError(
            f"Unknown policy mode {mode_name!r}; expected 'inference' or 'stochastic'."
        )

    # ---------------------------------------------------------- validation

    def _validation_steps(self) -> int:
        """Env steps per validated policy mode; 0 disables validation."""
        if (
            self.cfg.validation_interval_transitions is None
            and self.cfg.validation_interval <= 0
        ):
            return 0
        if self.cfg.validation_steps_per_env == 0:
            return 0
        if self.episode_evaluator is not None:
            return int(self.env.max_episode_length)
        if self.cfg.validation_steps_per_env is None:
            return 5 * int(self.cfg.steps_per_env_per_cycle)
        return int(self.cfg.validation_steps_per_env)

    def _should_validate(self) -> bool:
        """Whether this cycle ends in a validation.

        A transition-based cadence wins when one is configured: it is what
        makes two algorithms that collect at different rates comparable.
        """
        if self._validation_steps() <= 0:
            return False
        interval = self.cfg.validation_interval_transitions
        if interval is None:
            return self.current_iteration % int(self.cfg.validation_interval) == 0
        if self.global_timestep < self._next_validation_transitions:
            return False
        # Skip past any interval the last cycle jumped over in one rollout.
        while self._next_validation_transitions <= self.global_timestep:
            self._next_validation_transitions += int(interval)
        return True

    def _validate(self) -> None:
        """Score both policy modes, and compare features against the expert."""
        num_steps_per_env = self._validation_steps()
        count = self._runtime_ctx.num_envs
        waves = (self.cfg.validation_episodes + count - 1) // count
        reserve = (
            2 * count * num_steps_per_env * (waves if self.episode_evaluator else 1)
        )
        if not self._fits_interaction_budget(reserve):
            self._emit("validation/skipped", {"Data/evaluation_skipped_budget": 1.0})
            return
        self.rl_alg.eval_mode()
        try:
            payload: dict[str, float] = {}
            detailed: dict[str, float] = {}
            expert_mean = self._expert_feature_means()
            for mode_name in ("inference", "stochastic"):
                if self.episode_evaluator is None:
                    rollout = self._rollout(
                        self._action_fn(mode_name),
                        num_steps_per_env=num_steps_per_env,
                        from_reset=True,
                    )
                else:
                    rollout = self._episode_validation(mode_name)
                mode_payload, mode_detail = self._mode_metrics(
                    mode_name, rollout, expert_mean
                )
                payload.update(mode_payload)
                detailed.update(mode_detail)
            payload.update(self.rl_alg.eval_metrics())
            payload.update(
                self._sample_efficiency.update(
                    simulator_transitions=self.environment_transitions,
                    environment_return_mean=payload["Validation/return_mean"],
                    success_rate=payload.get("Validation/success_rate"),
                    environment_return_per_step=payload.get(
                        "Validation/return_per_step"
                    ),
                )
            )
            if self.reward_learner is not None:
                expert_success = self.reward_learner.expert_success_rate
                if expert_success is not None:
                    payload["Summary/expert_success_rate"] = float(expert_success)
            payload[TRANSITIONS] = float(self.global_timestep)
            self._emit("validation/summary", payload)
            if detailed:
                self._emit("validation/features", detailed)
        finally:
            self.rl_alg.train_mode()
            self._restart_collection()

    def _episode_validation(self, mode_name: str) -> _Rollout:
        """Score completed episodes, retaining pre-reset features and masks."""
        if self.episode_evaluator is None:
            raise RuntimeError("No episode evaluator configured.")
        batch = self.episode_evaluator(mode_name, self.cfg.validation_episodes)
        self.environment_transitions += batch.environment_transitions
        self.evaluation_transitions += batch.environment_transitions
        self.evaluation_episodes += batch.episode_count
        names = self.feature_source.spec.names or ()
        success = episode_success(
            batch.transitions.features,
            names,
            batch.mask,
            terminated=batch.transitions.terminated,
            truncated=batch.transitions.truncated,
        )
        return _Rollout(
            batch.transitions.features,
            batch.transitions.terminated | batch.transitions.truncated,
            batch.returns,
            success,
            batch.mask,
        )

    def _fits_interaction_budget(self, additional: int) -> bool:
        budget = self.cfg.total_transition_budget
        return budget is None or self.environment_transitions + additional <= budget

    def _mode_metrics(
        self,
        mode_name: str,
        rollout: "_Rollout",
        expert_mean: torch.Tensor | None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Summary and per-feature detail for one validated policy mode."""
        deterministic = mode_name == "inference"
        prefix = "Validation" if deterministic else f"Validation/{mode_name}"
        total_reward = rollout.total_reward
        payload = {
            f"{prefix}/return_mean": float(total_reward.mean().item()),
            f"{prefix}/return_std": float(total_reward.std(unbiased=False).item()),
        }
        if rollout.episode_mask is not None:
            payload[f"{prefix}/return_per_step"] = float(
                total_reward.sum() / rollout.episode_mask.sum()
            )
            payload[f"{prefix}/episode_length_mean"] = float(
                rollout.episode_mask.sum(1).float().mean()
            )
        if rollout.ever_success is not None:
            success_rate = float(rollout.ever_success.float().mean().item())
            payload[f"{prefix}/success_rate"] = success_rate
            summary_key = (
                "Summary/success_rate"
                if deterministic
                else "Summary/success_rate_stochastic"
            )
            payload[summary_key] = success_rate

        detailed: dict[str, float] = {}
        if expert_mean is None or self.reward_learner is None:
            return payload, detailed

        imitator_mean = self._imitator_feature_means(rollout).to(expert_mean)
        gap = expert_mean - imitator_mean
        if deterministic:
            payload["Summary/feature_gap_norm"] = float(gap.norm().item())
        names = self._feature_names()
        detailed.update(
            _named(f"ValidationFeatures/{mode_name}/imitator", names, imitator_mean)
        )
        detailed.update(_named(f"ValidationFeatures/{mode_name}/gap", names, gap))
        return payload, detailed

    def _expert_feature_means(self) -> torch.Tensor | None:
        if self.reward_learner is None:
            return None
        return self.reward_learner.sample_expert_feature_return_mean(self.device)

    def _imitator_feature_means(self, rollout: "_Rollout") -> torch.Tensor:
        """Per-feature discounted returns of a validation rollout. [D].

        Scored over the episodes the rollout contains rather than over the
        whole window: the expert side of this comparison is whole
        demonstrations, so accumulating an imitator return across a reset would
        compare two different quantities.
        """
        assert self.reward_learner is not None
        features = self._reward_feature_columns(rollout.features)
        if rollout.episode_mask is not None:
            return self.reward_learner.per_feature_return_mean(
                features, rollout.episode_mask
            )
        scratch = FeatureTrajectoryBuffer(
            replace(self.cfg.imitator_buffer, min_ep_len=1),
            feature_dim=self._reward_feature_dim,
            gamma=float(self.reward_learner.gamma),
            device=self.device,
        )
        scratch.add_rollout(features, rollout.dones)
        feats, mask, _ = scratch.sample_episodes(
            batch_size=int(self.reward_learner.batch_size), device=self.device
        )
        return self.reward_learner.per_feature_return_mean(feats, mask)

    def _feature_names(self) -> list[str]:
        """Labels for the per-feature validation detail, in reward-learner order."""
        if self.reward_features is not None:
            return list(self.reward_features.names)
        names = self.feature_source.spec.names
        if names is not None:
            return list(names)
        return [f"feat_{index}" for index in range(self._runtime_ctx.feature_dim)]

    # ------------------------------------------------------------- logging

    def _log_iteration(self) -> None:
        payload = dict(self._last_metrics)
        payload.update(training_summary(payload))
        payload[TRANSITIONS] = float(self.global_timestep)
        self._emit("iteration_summary", payload)

    def _emit(self, kind: str, payload: dict[str, float]) -> None:
        payload = dict(payload)
        payload.update(
            {
                "Summary/environment_transitions": float(self.environment_transitions),
                "Summary/evaluation_transitions": float(self.evaluation_transitions),
                "Summary/imitator_transitions": float(self.imitator_transitions),
                "Summary/training_episodes": float(self.training_episodes),
                "Summary/evaluation_episodes": float(self.evaluation_episodes),
            }
        )
        self.logger.log(
            LogEvent(
                kind=kind,
                step=self.global_timestep,
                iteration=self.current_iteration - 1,
                payload=payload,
            )
        )

    def finish_logger(self, exit_code: int = 0) -> None:
        self.logger.finish(exit_code=exit_code)

    # --------------------------------------------------------- checkpoints

    def _maybe_save_checkpoint(self) -> None:
        if self.log_dir is None:
            return
        if self.current_iteration % int(self.cfg.save_interval) != 0:
            return
        self.save(os.path.join(self.log_dir, f"model_{self.current_iteration}.pt"))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload: dict[str, Any] = {
            "iteration": int(self.current_iteration),
            "global_timestep": int(self.global_timestep),
            "rl_alg_state": self.rl_alg.save_state(),
            "interaction_counts": {
                name: getattr(self, name)
                for name in (
                    "environment_transitions",
                    "evaluation_transitions",
                    "imitator_transitions",
                    "training_episodes",
                    "evaluation_episodes",
                )
            },
        }
        if self.reward_learner is not None:
            payload["reward_learner_state"] = self.reward_learner.save_state()
        torch.save(payload, path)

    def load(self, path: str, *, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if "interaction_counts" not in checkpoint:
            raise ValueError(
                "Checkpoint has no interaction_counts; its evaluation cost is unknown. "
                "Use --init_policy_checkpoint for a new run instead of resuming its budget."
            )
        self.rl_alg.load_state(
            checkpoint["rl_alg_state"], load_optimizer=load_optimizer
        )
        if self.reward_learner is not None:
            self.reward_learner.load_state(
                checkpoint["reward_learner_state"], load_optimizer=load_optimizer
            )
        self.current_iteration = int(checkpoint["iteration"])
        self.global_timestep = int(checkpoint["global_timestep"])
        for name in (
            "environment_transitions",
            "evaluation_transitions",
            "imitator_transitions",
            "training_episodes",
            "evaluation_episodes",
        ):
            setattr(self, name, int(checkpoint["interaction_counts"][name]))

    def train_mode(self) -> None:
        self.rl_alg.train_mode()
        if self.reward_learner is not None:
            self.reward_learner.train_mode()

    def eval_mode(self) -> None:
        self.rl_alg.eval_mode()
        if self.reward_learner is not None:
            self.reward_learner.eval_mode()


def _named(prefix: str, names: list[str], values: torch.Tensor) -> dict[str, float]:
    """Build ``{prefix/name: value}`` from a [D] tensor."""
    return {
        f"{prefix}/{name}": float(values[index].item())
        for index, name in enumerate(names)
    }
