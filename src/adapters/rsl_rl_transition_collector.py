"""Environment-boundary transition collection shared by SAC and PMD."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from adapters.action_diagnostics import bounded_action_metrics
from adapters.rsl_rl_pre_reset_tap import IsaacPreResetTap
from algorithms.state_dependent_tanh_actor import (
    TanhPolicyProtocol,
    policy_observation_tensor,
)
from storage.observations import (
    Observation,
    critic_observation,
    detached_observation,
    stack_observations,
    where_observation,
)
from storage.transition_replay import TransitionRollout


class TransitionCollector:
    """Collect chunks that keep the true successor of every transition."""

    def __init__(
        self,
        env: Any,
        policy: TanhPolicyProtocol,
        feature_tap: IsaacPreResetTap,
        *,
        rollout_steps: int,
        device: str | torch.device,
        seed: int,
        action_bound: float = 1.0,
    ) -> None:
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be > 0.")
        if action_bound <= 0.0:
            raise ValueError("action_bound must be > 0.")
        self.env = env
        self.policy = policy
        self.feature_tap = feature_tap
        self.rollout_steps = int(rollout_steps)
        self.device = torch.device(device)
        self.num_envs = int(env.num_envs)
        self.action_bound = float(action_bound)
        environment_clip = getattr(env, "clip_actions", None)
        if environment_clip is not None and float(environment_clip) < action_bound:
            raise ValueError(
                "Environment action clip is narrower than policy action bound: "
                f"{environment_clip} < {action_bound}."
            )
        self._action_generator = torch.Generator(device=self.device).manual_seed(
            seed + 73
        )
        self._observations: Observation | None = None
        self._rollout_id = 0
        self._verified_terminal_observations = False

    def _get_observations(self) -> Observation:
        observations = self.env.get_observations()
        if isinstance(observations, tuple):
            observations = observations[0]
        return detached_observation(observations, self.device)

    def reset_state(self, *, reset_environment: bool) -> None:
        """Refresh observations after validation/video or checkpoint resume."""
        if reset_environment:
            self.env.reset()
        self._observations = self._get_observations()

    def collect(
        self,
        *,
        policy_version: int,
    ) -> tuple[TransitionRollout, dict[str, float]]:
        """Collect one chunk of ``[N,T,...]`` transitions with true successors."""
        if self._observations is None:
            self.reset_state(reset_environment=True)
        if self._observations is None:
            raise RuntimeError("Collector observations were not initialized.")
        observations = self._observations
        policy_steps: list[Observation] = []
        next_policy_steps: list[Observation] = []
        critic_steps: list[torch.Tensor] = []
        next_critic_steps: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        environment_rewards: list[torch.Tensor] = []
        terminated_steps: list[torch.Tensor] = []
        truncated_steps: list[torch.Tensor] = []
        env_device = torch.device(getattr(self.env, "device", self.device))

        with torch.no_grad():
            for _ in range(self.rollout_steps):
                action, _ = self.policy.sample_with_log_probability(
                    observations,
                    generator=self._action_generator,
                )
                if torch.any(action.abs() > self.action_bound + 1.0e-6):
                    raise ValueError("Tanh policy produced an out-of-bound action.")
                next_raw, reward, dones, extras = self.env.step(action.to(env_device))
                reset_observations = detached_observation(next_raw, self.device)
                done = (
                    torch.as_tensor(dones, device=self.device)
                    .reshape(-1)
                    .to(torch.bool)
                    .clone()
                )
                truncated = self._truncation_mask(extras, done)
                terminated = done & ~truncated
                successor_observations = self._successor_observations(
                    reset_observations, truncated=truncated, done=done
                )
                step_features = self.feature_tap.features()
                captured_reward = self.feature_tap.environment_reward()
                reward_tensor = (
                    torch.as_tensor(reward, device=self.device).reshape(-1).clone()
                )
                if not torch.allclose(
                    reward_tensor,
                    captured_reward,
                    atol=1.0e-6,
                    rtol=1.0e-6,
                ):
                    raise ValueError(
                        "Reward feature tap and env.step returned different rewards."
                    )
                policy_steps.append(observations)
                next_policy_steps.append(successor_observations)
                critic_steps.append(
                    critic_observation(observations).flatten(start_dim=1)
                )
                next_critic_steps.append(
                    critic_observation(successor_observations).flatten(start_dim=1)
                )
                actions.append(action.detach())
                features.append(step_features.detach())
                environment_rewards.append(reward_tensor.detach())
                terminated_steps.append(terminated)
                truncated_steps.append(truncated)
                observations = reset_observations

        self._observations = observations
        stacked_actions = torch.stack(actions, dim=1)
        terminated = torch.stack(terminated_steps, dim=1)
        truncated = torch.stack(truncated_steps, dim=1)
        rollout = TransitionRollout(
            policy_observations=stack_observations(policy_steps, dim=1),
            next_policy_observations=stack_observations(next_policy_steps, dim=1),
            critic_observations=torch.stack(critic_steps, dim=1),
            next_critic_observations=torch.stack(next_critic_steps, dim=1),
            actions=stacked_actions,
            features=torch.stack(features, dim=1),
            environment_rewards=torch.stack(environment_rewards, dim=1),
            terminated=terminated,
            truncated=truncated,
            # Truncated transitions carry the true terminal successor, so only a
            # genuine termination stops the Bellman backup.
            bootstrap_valid=~terminated,
            policy_version=int(policy_version),
            rollout_id=self._rollout_id,
        )
        self._rollout_id += 1
        metrics = bounded_action_metrics(rollout.actions, bound=self.action_bound)
        metrics["Data/rollout_steps_per_env"] = float(self.rollout_steps)
        metrics["Data/terminated_fraction"] = float(terminated.float().mean().item())
        metrics["Data/truncated_fraction"] = float(truncated.float().mean().item())
        metrics["Data/bootstrap_disabled_fraction"] = float(
            (~rollout.bootstrap_valid).float().mean().item()
        )
        return rollout, metrics

    def _truncation_mask(self, extras: Any, done: torch.Tensor) -> torch.Tensor:
        """Resolve the time-limit mask [N] reported alongside ``done``."""
        time_outs = extras.get("time_outs") if isinstance(extras, Mapping) else None
        if time_outs is None:
            time_outs = getattr(
                getattr(self.env, "unwrapped", self.env),
                "reset_time_outs",
                None,
            )
        if time_outs is None:
            return torch.zeros_like(done)
        return (
            torch.as_tensor(time_outs, device=self.device)
            .reshape(-1)
            .to(torch.bool)
            .clone()
            & done
        )

    def _successor_observations(
        self,
        reset_observations: Observation,
        *,
        truncated: torch.Tensor,
        done: torch.Tensor,
    ) -> Observation:
        """Replace post-reset observations by pre-reset ones where truncated."""
        if not bool(truncated.any()):
            return reset_observations
        terminal_observations = self.feature_tap.terminal_observations()
        if terminal_observations is None:
            raise RuntimeError(
                "Episodes timed out but the pre-reset tap captured no terminal "
                "observation; construct the tap with "
                "capture_terminal_observations=True."
            )
        self._verify_terminal_observations(
            terminal_observations, reset_observations, done
        )
        return where_observation(truncated, terminal_observations, reset_observations)

    def _verify_terminal_observations(
        self,
        terminal_observations: Observation,
        reset_observations: Observation,
        done: torch.Tensor,
    ) -> None:
        """Check once that the extra observation pass has no side effects.

        Environments that did not reset must observe the same state whether it
        is computed inside the reward manager or after it. A mismatch means the
        observation pipeline is stateful (noise, history buffers), in which case
        the captured terminal observation cannot be trusted.
        """
        if self._verified_terminal_observations:
            return
        surviving = ~done
        if not bool(surviving.any()):
            return
        for resolve in (critic_observation, policy_observation_tensor):
            captured = resolve(terminal_observations)[surviving]
            returned = resolve(reset_observations)[surviving]
            if not torch.allclose(captured, returned, atol=1.0e-5, rtol=1.0e-5):
                largest = float((captured - returned).abs().max().item())
                raise ValueError(
                    "Pre-reset observations differ from the observations the "
                    f"environment returned (max |difference| {largest:.3e}). The "
                    "observation pipeline is stateful, so terminal observations "
                    "cannot be captured this way."
                )
        self._verified_terminal_observations = True

    def save_state(self) -> dict[str, Any]:
        return {
            "action_generator_state": self._action_generator.get_state(),
            "rollout_id": self._rollout_id,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._action_generator.set_state(state["action_generator_state"].cpu())
        self._rollout_id = int(state["rollout_id"])
        self._observations = None
