"""Environment-boundary transition collection shared by SAC and PMD."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange

from adapters.action_diagnostics import bounded_action_metrics
from adapters.pre_reset_tap import IsaacPreResetTap
from algorithms.off_policy.actor import (
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


@dataclass(frozen=True, slots=True)
class _CollectedStep:
    """One environment step, in the fields a rollout stacks it into.

    ``reset_observations`` is what the environment handed back, which is where
    the next step starts; ``next_policy_observations`` is the true successor
    the Bellman backup needs, and the two differ exactly on the environments
    that a time limit reset.
    """

    policy_observations: Observation
    next_policy_observations: Observation
    actions: torch.Tensor
    features: torch.Tensor
    environment_rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    reset_observations: Observation


def _flat_critic(observations: Observation) -> torch.Tensor:
    """Resolve the critic's view and flatten everything after the time axis."""
    return rearrange(critic_observation(observations), "n t ... -> n t (...)")


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
        is_finite_horizon: bool = False,
    ) -> None:
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be > 0, got {rollout_steps}.")
        # The bound belongs to the policy that produces the actions; taking a
        # second copy is how a check and the thing it checks drift apart.
        action_bound = float(policy.model.action_bound)
        self.env = env
        self.policy = policy
        self.feature_tap = feature_tap
        self.rollout_steps = int(rollout_steps)
        self.device = torch.device(device)
        self.num_envs = int(env.num_envs)
        self.is_finite_horizon = bool(is_finite_horizon)
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
        self.env_device = torch.device(env.device)
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
        steps: list[_CollectedStep] = []
        with torch.no_grad():
            for _ in range(self.rollout_steps):
                steps.append(self.collect_step())

        rollout = self.stack_steps(steps, policy_version=policy_version)
        return rollout, self.rollout_metrics(rollout)

    @torch.no_grad()
    def collect_step(
        self, action_fn: Callable[[Observation], torch.Tensor] | None = None
    ) -> _CollectedStep:
        """Capture one vector step, advancing cached observations exactly once."""
        if self._observations is None:
            self.reset_state(reset_environment=True)
        step = self._step_once(self._observations, action_fn=action_fn)
        self._observations = step.reset_observations
        return step

    def invalidate_observations(self) -> None:
        """Forget state without resetting or stepping the environment."""
        self._observations = None

    def _step_once(
        self, observations: Observation, *, action_fn=None
    ) -> "_CollectedStep":
        """Act once, step the env, and capture the transition that produced."""
        if action_fn is None:
            action, _ = self.policy.sample_with_log_probability(
                observations, generator=self._action_generator
            )
        else:
            action = action_fn(observations)
        largest_action = float(action.abs().max().item())
        if largest_action > self.action_bound + 1.0e-6:
            raise ValueError(
                f"Tanh policy produced an action of {largest_action:.4f}, outside "
                f"its bound {self.action_bound:.4f}."
            )
        next_raw, reward, dones, extras = self.env.step(action.to(self.env_device))
        reset_observations = detached_observation(next_raw, self.device)
        done = self._as_env_flags(dones, name="dones")
        truncated = self._truncation_mask(extras, done)
        # Isaac preserves both flags, including a failure on the timeout step.
        native = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        if hasattr(native, "reset_terminated"):
            terminated = self._as_env_flags(
                native.reset_terminated, name="reset_terminated"
            )
        else:
            terminated = done & ~truncated
        step_features = self.feature_tap.features()
        reward_tensor = torch.as_tensor(reward, device=self.device).reshape(-1).clone()
        self._require_matching_reward(reward_tensor)
        return _CollectedStep(
            policy_observations=observations,
            next_policy_observations=self._successor_observations(
                reset_observations, done=done
            ),
            actions=action.detach(),
            features=step_features.detach(),
            environment_rewards=reward_tensor.detach(),
            terminated=terminated,
            truncated=truncated,
            reset_observations=reset_observations,
        )

    def stack_steps(
        self, steps: list["_CollectedStep"], *, policy_version: int
    ) -> TransitionRollout:
        """Stack per-step records into one ``[N,T,...]`` chunk."""

        def stack(name: str) -> torch.Tensor:
            return torch.stack([getattr(step, name) for step in steps], dim=1)

        policy_observations = stack_observations(
            [step.policy_observations for step in steps], dim=1
        )
        next_policy_observations = stack_observations(
            [step.next_policy_observations for step in steps], dim=1
        )
        terminated = stack("terminated")
        truncated = stack("truncated")
        rollout = TransitionRollout(
            policy_observations=policy_observations,
            next_policy_observations=next_policy_observations,
            critic_observations=_flat_critic(policy_observations),
            next_critic_observations=_flat_critic(next_policy_observations),
            actions=stack("actions"),
            features=stack("features"),
            environment_rewards=stack("environment_rewards"),
            terminated=terminated,
            truncated=truncated,
            bootstrap_valid=~(terminated | (truncated & self.is_finite_horizon)),
            policy_version=int(policy_version),
            rollout_id=self._rollout_id,
        )
        self._rollout_id += 1
        return rollout

    def rollout_metrics(self, rollout: TransitionRollout) -> dict[str, float]:
        metrics = bounded_action_metrics(rollout.actions, bound=self.action_bound)
        metrics["Data/rollout_steps_per_env"] = float(self.rollout_steps)
        metrics["Data/terminated_fraction"] = float(
            rollout.terminated.float().mean().item()
        )
        metrics["Data/truncated_fraction"] = float(
            rollout.truncated.float().mean().item()
        )
        metrics["Data/bootstrap_disabled_fraction"] = float(
            (~rollout.bootstrap_valid).float().mean().item()
        )
        return metrics

    def _require_matching_reward(self, step_reward: torch.Tensor) -> None:
        """The features must reconstruct the reward the environment returned.

        They come from two different places - the tap inside the reward manager
        and env.step's return - so a mismatch means the features being learned
        from are not the reward being collected against.
        """
        captured = self.feature_tap.environment_reward()
        if torch.allclose(step_reward, captured, atol=1.0e-6, rtol=1.0e-6):
            return
        largest = float((step_reward - captured).abs().max().item())
        raise ValueError(
            "Reward feature tap and env.step returned different rewards: "
            f"maximum difference {largest:.3e}."
        )

    def _as_env_flags(self, value: Any, *, name: str) -> torch.Tensor:
        """Normalize a per-environment flag vector to ``[N]`` bool."""
        flags = (
            torch.as_tensor(value, device=self.device)
            .reshape(-1)
            .to(torch.bool)
            .clone()
        )
        if tuple(flags.shape) != (self.num_envs,):
            raise ValueError(
                f"Expected {name} of shape ({self.num_envs},), got {tuple(flags.shape)}."
            )
        return flags

    def _truncation_mask(self, extras: Any, done: torch.Tensor) -> torch.Tensor:
        """Resolve the time-limit mask [N] reported alongside ``done``.

        Without it every timeout would be recorded as a termination and its
        bootstrap silently dropped, which is a bias, not a missing metric - so
        an environment that reports neither source is an error, not a default.
        """
        if isinstance(extras, Mapping) and "time_outs" in extras:
            return self._as_env_flags(extras["time_outs"], name="time_outs") & done
        unwrapped = getattr(self.env, "unwrapped", self.env)
        if hasattr(unwrapped, "reset_time_outs"):
            return (
                self._as_env_flags(unwrapped.reset_time_outs, name="reset_time_outs")
                & done
            )
        raise ValueError(
            "The environment reports no time limits: expected extras['time_outs'] "
            f"or env.reset_time_outs, got extras keys "
            f"{sorted(extras) if isinstance(extras, Mapping) else type(extras).__name__}."
        )

    def _successor_observations(
        self,
        reset_observations: Observation,
        *,
        done: torch.Tensor,
    ) -> Observation:
        """Keep true successors for every ending, including non-bootstrapped ones."""
        if not bool(done.any()):
            return reset_observations
        terminal_observations = self.feature_tap.terminal_observations()
        if terminal_observations is None:
            raise RuntimeError(
                "Episodes ended but the pre-reset tap captured no terminal "
                "observation; construct the tap with "
                "capture_terminal_observations=True."
            )
        self._verify_terminal_observations(
            terminal_observations, reset_observations, done
        )
        return where_observation(done, terminal_observations, reset_observations)

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
