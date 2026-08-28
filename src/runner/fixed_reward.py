"""Runner for fixed-reward off-policy RL without IRL or BC state."""

from __future__ import annotations

import os
from typing import Any, Callable

import torch

from config.fixed_reward import FixedRewardRunnerCfg
from interfaces import FixedRewardRlAlgorithm
from runner.loggers import LogEvent, Logger, NoopLogger
from runner.overview import SampleEfficiencyTracker, training_overview_metrics


class FixedRewardRunner:
    """Train one fixed-reward RL algorithm and report task performance."""

    def __init__(
        self,
        *,
        env: Any,
        rl_algorithm: FixedRewardRlAlgorithm,
        cfg: FixedRewardRunnerCfg,
        device: str | torch.device,
        log_dir: str | None,
        success_fn: Callable[[Any], torch.Tensor] | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.env = env
        self.rl_algorithm = rl_algorithm
        self.cfg = cfg
        self.device = torch.device(device)
        self.log_dir = log_dir
        self.success_fn = success_fn
        self.logger = logger or NoopLogger()
        self.current_iteration = 0
        self.global_timestep = 0
        self._last_validation_timestep: int | None = None
        self._sample_efficiency = SampleEfficiencyTracker(
            validation_steps=cfg.validation_steps,
            return_per_step_thresholds=cfg.return_per_step_thresholds,
            success_rate_thresholds=cfg.success_rate_thresholds,
        )

    def learn(self, num_iterations: int) -> None:
        self.logger.start(num_iterations)
        if self.cfg.validate_before_training:
            self._validate()
        for _ in range(num_iterations):
            self._train_iteration()
        self._save_final_checkpoint()

    def learn_to_transition_budget(self, transition_budget: int) -> None:
        """Train until the next indivisible rollout reaches ``transition_budget``."""
        if transition_budget <= 0:
            raise ValueError("transition_budget must be > 0.")
        self.logger.start(num_iterations=0)
        if self.cfg.validate_before_training:
            self._validate()
        next_validation = self.cfg.validation_interval_transitions
        while self.global_timestep < transition_budget:
            previous_timestep = self.global_timestep
            self._train_iteration()
            if next_validation is not None:
                if previous_timestep < next_validation <= self.global_timestep:
                    self._validate()
                while next_validation <= self.global_timestep:
                    next_validation += self.cfg.validation_interval_transitions or 1
        if self._last_validation_timestep != self.global_timestep:
            self._validate()
        self._save_final_checkpoint()

    def _train_iteration(self) -> None:
        self.rl_algorithm.train_mode()
        metrics = self.rl_algorithm.update()
        self.global_timestep += int(self.rl_algorithm.simulator_steps_per_update)
        payload = dict(metrics)
        payload.update(self.rl_algorithm.train_metrics())
        payload.update(training_overview_metrics(payload))
        payload["Data/simulator_transitions_total"] = float(self.global_timestep)
        self.logger.log(
            LogEvent(
                kind="iteration_summary",
                step=self.global_timestep,
                iteration=self.current_iteration,
                payload=payload,
            )
        )
        self.current_iteration += 1
        if (
            self.cfg.validation_interval_transitions is None
            and self.cfg.validation_interval > 0
            and self.current_iteration % self.cfg.validation_interval == 0
        ):
            self._validate()
        if (
            self.log_dir is not None
            and self.current_iteration % self.cfg.save_interval == 0
        ):
            self.save(os.path.join(self.log_dir, f"model_{self.current_iteration}.pt"))

    def _save_final_checkpoint(self) -> None:
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_iteration}.pt"))

    def _validate(self) -> None:
        self.rl_algorithm.eval_mode()
        self.env.reset()
        observations = self.env.get_observations()
        if isinstance(observations, tuple):
            observations = observations[0]
        total_reward = torch.zeros(int(self.env.num_envs), device=self.device)
        ever_success = torch.zeros(
            int(self.env.num_envs), dtype=torch.bool, device=self.device
        )
        with torch.no_grad():
            for _ in range(self.cfg.validation_steps):
                actions = self.rl_algorithm.actor.act_inference(observations)
                observations, rewards, _, _ = self.env.step(actions.to(self.env.device))
                total_reward += torch.as_tensor(rewards, device=self.device).reshape(-1)
                if self.success_fn is not None:
                    success = (
                        torch.as_tensor(self.success_fn(self.env), device=self.device)
                        .reshape(-1)
                        .to(torch.bool)
                    )
                    ever_success |= success
        environment_return_mean = float(total_reward.mean().item())
        success_rate = (
            float(ever_success.float().mean().item())
            if self.success_fn is not None
            else None
        )
        payload = {
            "Validation/environment_return_mean": environment_return_mean,
            "Validation/environment_return_std": float(
                total_reward.std(unbiased=False).item()
            ),
            "Data/simulator_transitions_total": float(self.global_timestep),
        }
        if success_rate is not None:
            payload["Validation/success_rate"] = success_rate
        payload.update(
            self._sample_efficiency.update(
                simulator_transitions=self.global_timestep,
                environment_return_mean=environment_return_mean,
                success_rate=success_rate,
            )
        )
        self.logger.log(
            LogEvent(
                kind="validation/summary",
                step=self.global_timestep,
                iteration=self.current_iteration - 1,
                payload=payload,
                label="inference",
            )
        )
        self._last_validation_timestep = self.global_timestep
        self.rl_algorithm.reset_collection_state()

    def record_final_rollouts(
        self,
        *,
        recorder: Any,
        video_length: int,
        mode_names: tuple[str, ...] = ("inference", "stochastic"),
    ) -> None:
        """Record deterministic and stochastic target-policy videos."""
        if video_length <= 0:
            return
        valid_modes = {"inference", "stochastic"}
        unknown_modes = set(mode_names) - valid_modes
        if unknown_modes:
            raise ValueError(
                f"Unknown video modes {sorted(unknown_modes)}; "
                f"expected a subset of {sorted(valid_modes)}."
            )

        self.rl_algorithm.eval_mode()
        try:
            for mode_name in mode_names:
                self.rl_algorithm.reset_collection_state()
                observations = self.env.get_observations()
                if isinstance(observations, tuple):
                    observations = observations[0]
                recorder.start_recording(f"final_{mode_name}")
                try:
                    for _ in range(video_length):
                        if mode_name == "inference":
                            actions = self.rl_algorithm.actor.act_inference(
                                observations
                            )
                        else:
                            actions = self.rl_algorithm.actor.act(observations)
                        observations, _, _, _ = self.env.step(
                            actions.to(self.env.device)
                        )
                finally:
                    if getattr(recorder, "recording", False):
                        recorder.stop_recording()
        finally:
            self.rl_algorithm.train_mode()
            self.rl_algorithm.reset_collection_state()

    def save(self, path: str) -> None:
        torch.save(
            {
                "iter": self.current_iteration,
                "global_timestep": self.global_timestep,
                "rl_alg_state": self.rl_algorithm.save_state(),
            },
            path,
        )

    def load(self, path: str, *, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.current_iteration = int(checkpoint["iter"])
        self.global_timestep = int(checkpoint["global_timestep"])
        self.rl_algorithm.load_state(
            checkpoint["rl_alg_state"], load_optimizer=load_optimizer
        )

    def finish_logger(self, exit_code: int = 0) -> None:
        self.logger.finish(exit_code=exit_code)
