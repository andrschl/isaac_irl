from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import torch

from interfaces import FeatureSource, FeatureStep, RlAlgorithm
from runner.console_logger import ConsoleLogger
from runner.loggers import LogEvent, Logger, NoopLogger
from storage.feature_storage import FeatureBufCfg, FeatureTrajectoryBuffer
from utils.runtime_context import RuntimeContext


@dataclass(slots=True)
class IrlRunnerCfg:
    """Runner/trainer-only config."""

    steps_per_env_per_cycle: int = 24
    save_interval: int = 50

    rl_updates_per_cycle: int = 1
    reward_updates_per_cycle: int = 1
    # Read by the training script when constructing the RL adapter; the
    # runner itself no longer routes the reward signal.
    use_learned_reward: bool = True

    imitator_buffer: FeatureBufCfg = field(default_factory=FeatureBufCfg)
    expert_buffer: FeatureBufCfg = field(
        default_factory=lambda: FeatureBufCfg(min_ep_len=1, store_device="cpu")
    )
    expert_num_envs: int = 1

    # Length of the dedicated full-trajectory imitator rollout used to refresh
    # the reward-learner imitator buffer between reward updates. None resolves
    # to ``env.max_episode_length`` at runtime so each env produces ~1 full
    # episode per cycle (matching expert demo length).
    imitator_rollout_steps_per_env: int | None = None

    # Run validation rollouts every this many learning iterations. 0 disables.
    validation_interval: int = 50
    # Number of validation env steps per policy mode. None = 5x training rollout.
    # 0 disables validation.
    validation_steps_per_env: int | None = None


Observation = Any
FeatureSink = Callable[[FeatureStep], None]
StepCallback = Callable[[], None]


class IrlRunner:
    """Algorithm-agnostic runner for policy + feature-buffer reward learning.

    The runner coordinates three things:
      1. The RL algorithm's own ``update()`` (which internally drives env
         collection — the runner does not step the env for RL collection).
      2. A dedicated imitator rollout that refreshes the reward learner's
         imitator buffer with [N, D] features from the current policy.
      3. The reward learner's update.

    The env contract (``step`` / ``get_observations``) appears in this file
    only for the imitator and validation rollouts. The RL collection path
    lives entirely inside ``rl_alg.update()``.
    """

    def __init__(
        self,
        env,
        *,
        rl_alg: RlAlgorithm,
        irl_alg: Any,
        feature_source: FeatureSource,
        runner_cfg: IrlRunnerCfg,
        success_fn: Callable[[Any], torch.Tensor] | None = None,
        log_dir: str | None = None,
        device: str | torch.device = "cpu",
        logger: Logger | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.env = env
        self.rl_alg = rl_alg
        self.irl_alg = irl_alg
        self.feature_source = feature_source
        self.success_fn = success_fn
        self.cfg = runner_cfg
        self.log_dir = log_dir
        self.logger: Logger = logger or NoopLogger()

        self.current_learning_iteration = 0
        self.global_timestep = 0
        self._last_rl_metrics: dict[str, float] = {"RL/policy_loss": float("nan")}
        self._last_reward_metrics: dict[str, float] = {
            "IRL/reward_loss": float("nan"),
            "IRL/feature_exp_diff_norm": float("nan"),
        }

        self._validate_cfg()

        self._runtime_ctx = RuntimeContext(
            num_envs=int(self.env.num_envs),
            feature_dim=int(feature_source.spec.dim),
            device=str(self.device),
        )

    def _validate_cfg(self) -> None:
        cfg = self.cfg
        if cfg.steps_per_env_per_cycle <= 0:
            raise ValueError(f"`steps_per_env_per_cycle` must be > 0, got {cfg.steps_per_env_per_cycle}.")
        if cfg.save_interval <= 0:
            raise ValueError(f"`save_interval` must be > 0, got {cfg.save_interval}.")
        if cfg.rl_updates_per_cycle <= 0:
            raise ValueError(f"`rl_updates_per_cycle` must be > 0, got {cfg.rl_updates_per_cycle}.")
        if cfg.reward_updates_per_cycle <= 0:
            raise ValueError(f"`reward_updates_per_cycle` must be > 0, got {cfg.reward_updates_per_cycle}.")
        if cfg.expert_num_envs <= 0:
            raise ValueError(f"`expert_num_envs` must be > 0, got {cfg.expert_num_envs}.")
        if cfg.imitator_rollout_steps_per_env is not None and int(cfg.imitator_rollout_steps_per_env) <= 0:
            raise ValueError(
                "`imitator_rollout_steps_per_env` must be > 0 or None, "
                f"got {cfg.imitator_rollout_steps_per_env}."
            )
        if cfg.validation_interval < 0:
            raise ValueError(f"`validation_interval` must be >= 0, got {cfg.validation_interval}.")
        if cfg.validation_steps_per_env is not None and int(cfg.validation_steps_per_env) < 0:
            raise ValueError(
                f"`validation_steps_per_env` must be >= 0 or null, got {cfg.validation_steps_per_env}."
            )

    def _get_obs(self):
        obs = self.env.get_observations()
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def _obs_to_device(self, obs: Observation) -> Observation:
        """Move tensor leaves or TensorDict-like observations to the runner device."""
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        to_device = getattr(obs, "to", None)
        if callable(to_device):
            return to_device(self.device)
        if isinstance(obs, dict):
            return {key: self._obs_to_device(value) for key, value in obs.items()}
        if isinstance(obs, tuple):
            return tuple(self._obs_to_device(value) for value in obs)
        if isinstance(obs, list):
            return [self._obs_to_device(value) for value in obs]
        return obs

    def _as_feature_tensor(self, features: Any) -> torch.Tensor:
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)
        return features.to(self.device)

    def _step_features(self) -> torch.Tensor:
        features = self._as_feature_tensor(self.feature_source.features(self.env))
        if features.ndim != 2:
            raise ValueError(f"Expected feature_source.features(env) shape [N, D], got {tuple(features.shape)}.")
        expected_shape = (int(self._runtime_ctx.num_envs), int(self._runtime_ctx.feature_dim))
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                f"Expected feature_source.features(env) shape {expected_shape}, got {tuple(features.shape)}."
            )
        return features

    def _rollout(
        self,
        obs: Any,
        *,
        num_steps_per_env: int,
        action_fn: Callable[[Any], torch.Tensor],
        feature_sink: FeatureSink | None = None,
        on_step: StepCallback | None = None,
    ) -> Any:
        """Roll out the environment for a fixed vectorized horizon.

        Used for imitator collection, validation, and final-video rollouts —
        all pure (no PPO storage side effects). The RL collection rollout is
        owned by ``rl_alg.update()`` and never passes through here.

        Shapes:
            actions:      [N, A]
            dones:        [N]
            features:     [N, D] when extracted
        """
        with torch.inference_mode():
            for _ in range(num_steps_per_env):
                actions = action_fn(obs)
                obs_next, _, dones, _ = self.env.step(actions.to(self.env.device))
                obs_next = self._obs_to_device(obs_next)
                done_vec = self._as_done_vector(dones)

                if feature_sink is not None:
                    features = self._step_features()
                    feature_sink(FeatureStep(features=features, dones=done_vec))

                if on_step is not None:
                    on_step()

                obs = obs_next
                self.global_timestep += int(self._runtime_ctx.num_envs)
        return obs

    def _as_done_vector(self, dones: torch.Tensor) -> torch.Tensor:
        """Normalize environment dones to a boolean vector [N] on runner device."""
        if not isinstance(dones, torch.Tensor):
            dones = torch.as_tensor(dones)
        done_vec = dones.to(device=self.device).reshape(-1).to(torch.bool)
        expected_shape = (int(self._runtime_ctx.num_envs),)
        if tuple(done_vec.shape) != expected_shape:
            raise ValueError(f"Expected dones shape {expected_shape}, got {tuple(dones.shape)}.")
        return done_vec

    def _resolved_imitator_rollout_steps(self) -> int:
        if self.cfg.imitator_rollout_steps_per_env is not None:
            return int(self.cfg.imitator_rollout_steps_per_env)
        max_ep = getattr(self.env, "max_episode_length", None)
        if max_ep is None:
            raise ValueError(
                "Could not infer imitator rollout length: env has no "
                "`max_episode_length`. Set `runner.imitator_rollout_steps_per_env` "
                "explicitly."
            )
        max_ep_int = int(max_ep)
        if max_ep_int <= 0:
            raise ValueError(
                f"env.max_episode_length must be > 0, got {max_ep_int}. "
                "Set `runner.imitator_rollout_steps_per_env` explicitly."
            )
        return max_ep_int

    def _collect_imitator_trajectories(self, obs: Any) -> Any:
        """Refresh reward-learner imitator data from a fresh policy rollout."""
        self.irl_alg.clear_imitator()
        obs = self._rollout(
            obs,
            num_steps_per_env=self._resolved_imitator_rollout_steps(),
            action_fn=self.rl_alg.actor.act,
            feature_sink=self.irl_alg.add_imitator,
        )
        self.irl_alg.finalize_imitator()
        return obs

    def _run_rl_updates(self) -> None:
        metrics: dict[str, float] = {}
        for _ in range(int(self.cfg.rl_updates_per_cycle)):
            metrics.update(self.rl_alg.update())
        if metrics:
            self._last_rl_metrics.update(metrics)

    def _run_reward_updates(self) -> None:
        if not self.irl_alg.can_update():
            return
        metrics: dict[str, float] = {}
        for _ in range(int(self.cfg.reward_updates_per_cycle)):
            metrics.update(self.irl_alg.update())
        if metrics:
            self._last_reward_metrics.update(metrics)

    def _emit(
        self,
        kind: str,
        payload: dict[str, float],
        *,
        iteration: int = 0,
        label: str = "",
        key_prefix: str = "",
    ) -> None:
        self.logger.log(
            LogEvent(
                kind=kind,
                step=self.global_timestep,
                iteration=iteration,
                payload=payload,
                label=label,
                key_prefix=key_prefix,
            )
        )

    @staticmethod
    def _scalar_payload(prefix: str, names: list[str], values: torch.Tensor) -> dict[str, float]:
        """Build a ``{prefix/name: float(value)}`` dict from a [D] tensor."""
        return {f"{prefix}/{name}": float(values[i].item()) for i, name in enumerate(names)}

    def _log_iteration(self, iteration: int) -> None:
        payload: dict[str, float] = {}
        payload.update(self._last_reward_metrics)
        payload.update(self._last_rl_metrics)
        payload.update(self.rl_alg.train_metrics())
        payload.update(self.irl_alg.train_metrics())
        self._emit("iteration_summary", payload, iteration=iteration)

    def _resolved_validation_steps(self) -> int:
        if self.cfg.validation_interval <= 0:
            return 0
        if self.cfg.validation_steps_per_env is None:
            return 5 * int(self.cfg.steps_per_env_per_cycle)
        return int(self.cfg.validation_steps_per_env)

    def _should_validate(self, learning_iteration: int) -> bool:
        validation_steps = self._resolved_validation_steps()
        return validation_steps > 0 and learning_iteration % int(self.cfg.validation_interval) == 0

    def _feature_names(self, feature_dim: int) -> list[str]:
        names = self.feature_source.spec.names
        if names is not None:
            return list(names)
        return [f"feat_{feature_idx}" for feature_idx in range(feature_dim)]

    def _make_success_accumulator(self) -> tuple[torch.Tensor | None, StepCallback | None]:
        """Build a per-env any-time success accumulator for validation rollouts.

        Returns ``(None, None)`` when no ``success_fn`` was configured. Otherwise
        returns a ``[N]`` bool tensor and a step callback that ORs the current
        env's success flags into it.
        """
        if self.success_fn is None:
            return None, None
        success_fn = self.success_fn
        ever_success = torch.zeros(self._runtime_ctx.num_envs, dtype=torch.bool, device=self.device)

        def _accumulate() -> None:
            step_success = torch.as_tensor(success_fn(self.env)).reshape(-1).to(
                device=ever_success.device, dtype=torch.bool
            )
            ever_success[step_success] = True

        return ever_success, _accumulate

    def _imitator_feature_expectations(
        self,
        obs: Any,
        *,
        action_fn: Callable[[Any], torch.Tensor],
        num_steps_per_env: int,
    ) -> tuple[Any, torch.Tensor | None, float | None]:
        """Roll out the policy (no training) and return (obs, per-feature expectations, success_rate)."""
        scratch = FeatureTrajectoryBuffer(
            cfg=replace(self.cfg.imitator_buffer, min_ep_len=1),
            ctx=self._runtime_ctx,
            gamma=float(self.irl_alg.gamma),
        )
        ever_success, on_step = self._make_success_accumulator()

        def _scratch_sink(step: FeatureStep) -> None:
            scratch.add_step(z=step.features, done=step.dones)

        obs = self._rollout(
            obs,
            num_steps_per_env=num_steps_per_env,
            action_fn=action_fn,
            feature_sink=_scratch_sink,
            on_step=on_step,
        )
        scratch.finalize_in_progress_episodes()

        success_rate = None if ever_success is None else float(ever_success.float().mean().item())
        if len(scratch) == 0:
            return obs, None, success_rate
        feats, mask, _ = scratch.sample_episodes(batch_size=int(self.irl_alg.batch_size), device=self.device)
        return obs, self.irl_alg.per_feature_return_mean(feats, mask), success_rate

    def _run_validation(self, obs: Any, iteration: int) -> Any:
        num_steps_per_env = self._resolved_validation_steps()
        if num_steps_per_env <= 0:
            return obs

        self.eval_mode()
        try:
            self._log_validation_eval_metrics(iteration)
            expert_mean, feature_names = self._log_validation_expert(iteration)
            if expert_mean is None:
                return obs
            actor = self.rl_alg.actor
            for mode_name, action_fn in (
                ("stochastic", actor.act),
                ("inference", actor.act_inference),
            ):
                obs = self._log_validation_mode(
                    obs,
                    iteration=iteration,
                    mode_name=mode_name,
                    action_fn=action_fn,
                    num_steps_per_env=num_steps_per_env,
                    expert_mean=expert_mean,
                    feature_names=feature_names,
                )
        finally:
            self.train_mode()
        return obs

    def _log_validation_eval_metrics(self, iteration: int) -> None:
        metrics = self.rl_alg.eval_metrics()
        if not metrics:
            return
        self._emit("validation/summary", metrics, iteration=iteration)

    def _log_validation_expert(self, iteration: int) -> tuple[torch.Tensor | None, list[str]]:
        expert_mean = self.irl_alg.sample_expert_feature_return_mean(self.device)
        if expert_mean is None:
            return None, []
        feature_names = self._feature_names(expert_mean.shape[0])

        expert_prefix = "ValidationFeatures/expert"
        expert_payload = self._scalar_payload(expert_prefix, feature_names, expert_mean)
        self._emit(
            "validation/features",
            expert_payload,
            iteration=iteration,
            label="group=expert",
            key_prefix=f"{expert_prefix}/",
        )

        expert_success_rate = self.irl_alg.expert_success_rate
        if expert_success_rate is not None:
            self._emit(
                "validation/expert_success_rate",
                {"Validation/expert/success_rate": float(expert_success_rate)},
                iteration=iteration,
            )
        return expert_mean, feature_names

    def _log_validation_mode(
        self,
        obs: Any,
        *,
        iteration: int,
        mode_name: str,
        action_fn: Callable[[Any], torch.Tensor],
        num_steps_per_env: int,
        expert_mean: torch.Tensor,
        feature_names: list[str],
    ) -> Any:
        obs, imitator_mean, success_rate = self._imitator_feature_expectations(
            obs, action_fn=action_fn, num_steps_per_env=num_steps_per_env,
        )

        summary: dict[str, float] = {}
        expert_success_rate = self.irl_alg.expert_success_rate
        if success_rate is not None:
            summary[f"Validation/{mode_name}/success_rate"] = success_rate
            if expert_success_rate is not None:
                summary[f"Validation/{mode_name}/success_rate_gap"] = (
                    float(expert_success_rate) - success_rate
                )

        if imitator_mean is not None:
            imitator_mean = imitator_mean.to(device=expert_mean.device, dtype=expert_mean.dtype)
            gap_mean = expert_mean - imitator_mean
            summary[f"Validation/{mode_name}/feature_exp_diff_norm"] = float(gap_mean.norm().item())

            imitator_prefix = f"ValidationFeatures/{mode_name}/imitator"
            gap_prefix = f"ValidationFeatures/{mode_name}/gap"
            imitator_payload = self._scalar_payload(imitator_prefix, feature_names, imitator_mean)
            gap_payload = self._scalar_payload(gap_prefix, feature_names, gap_mean)
            self._emit(
                "validation/features",
                imitator_payload,
                iteration=iteration,
                label=f"mode={mode_name} group=imitator",
                key_prefix=f"{imitator_prefix}/",
            )
            self._emit(
                "validation/features",
                gap_payload,
                iteration=iteration,
                label=f"mode={mode_name} group=gap",
                key_prefix=f"{gap_prefix}/",
            )

        self._emit(
            "validation/summary",
            summary,
            iteration=iteration,
            label=f"mode={mode_name}",
        )
        return obs

    def _maybe_save_checkpoint(self, iteration: int) -> None:
        if self.log_dir is None:
            return
        if (iteration + 1) % int(self.cfg.save_interval) != 0:
            return
        self.save(os.path.join(self.log_dir, f"model_{iteration + 1}.pt"))

    def finish_logger(self, exit_code: int = 0) -> None:
        self.logger.finish(exit_code=exit_code)

    def record_final_rollouts(
        self,
        *,
        recorder: Any,
        video_length: int,
        mode_names: tuple[str, ...] = ("inference", "stochastic"),
    ) -> None:
        """Drive the ``RecordVideo``-wrapped env for one clip per policy mode.

        ``mode_names`` selects which Actor methods to use: "inference" -> deterministic,
        "stochastic" -> sampled. Any other name raises.
        """
        if video_length <= 0:
            return
        actor = self.rl_alg.actor
        method_for_mode = {
            "inference": actor.act_inference,
            "stochastic": actor.act,
        }
        self.eval_mode()
        try:
            obs = self._obs_to_device(self._get_obs())
            for mode_name in mode_names:
                if mode_name not in method_for_mode:
                    raise ValueError(
                        f"Unknown mode_name {mode_name!r}; expected one of {sorted(method_for_mode)}."
                    )
                recorder.start_recording(f"final_{mode_name}")
                obs = self._rollout(
                    obs,
                    num_steps_per_env=video_length,
                    action_fn=method_for_mode[mode_name],
                )
                if getattr(recorder, "recording", False):
                    recorder.stop_recording()
        finally:
            self.train_mode()

    def train_mode(self) -> None:
        self.rl_alg.train_mode()
        self.irl_alg.train_mode()

    def eval_mode(self) -> None:
        self.rl_alg.eval_mode()
        self.irl_alg.eval_mode()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self.logger.start(int(num_learning_iterations))

        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf") and hasattr(self.env, "max_episode_length"):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        self.train_mode()

        start_iter = self.current_learning_iteration
        for it in range(start_iter, start_iter + int(num_learning_iterations)):
            self._run_rl_updates()
            obs = self._obs_to_device(self._get_obs())
            obs = self._collect_imitator_trajectories(obs)
            self._run_reward_updates()

            self.current_learning_iteration = it + 1
            self._log_iteration(iteration=it)

            if self._should_validate(it + 1):
                obs = self._run_validation(obs, it)
            self._maybe_save_checkpoint(it)

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload: dict[str, Any] = {
            "iter": int(self.current_learning_iteration),
            "global_timestep": int(self.global_timestep),
            "rl_alg_state": self.rl_alg.save_state(),
            "irl_alg_state": self.irl_alg.save_state(),
        }
        torch.save(payload, path)

    def load(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location=self.device)
        rl_alg_state = checkpoint.get("rl_alg_state")
        irl_alg_state = checkpoint.get("irl_alg_state")
        if rl_alg_state is None or irl_alg_state is None:
            raise ValueError(
                "Checkpoint must contain `rl_alg_state` and `irl_alg_state`. "
                "Old flat checkpoints are not supported by the adapter runner."
            )

        self.rl_alg.load_state(rl_alg_state, load_optimizer=load_optimizer)
        self.irl_alg.load_state(irl_alg_state, load_optimizer=load_optimizer)
        self.current_learning_iteration = int(checkpoint.get("iter", 0))
        self.global_timestep = int(checkpoint.get("global_timestep", 0))
        return checkpoint.get("infos", None)
