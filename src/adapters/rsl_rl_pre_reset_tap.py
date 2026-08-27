"""Capture of everything Isaac computes before its automatic environment reset.

``ManagerBasedRLEnv.step`` computes terminations, then rewards, then resets the
terminated environments and only afterwards computes the observations it
returns. Reward terms and the true terminal observation are therefore only
reachable from inside the reward-manager call, which this tap wraps.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import torch

from interfaces import FeatureSpec
from storage.observations import Observation, detached_observation


class IsaacPreResetTap:
    """Capture raw reward terms, the env reward, and terminal observations."""

    def __init__(
        self,
        env: Any,
        *,
        ignored_terms: Iterable[str] = (),
        device: str | torch.device | None = None,
        capture_terminal_observations: bool = False,
    ) -> None:
        self.env = env
        self.unwrapped = getattr(env, "unwrapped", env)
        self.reward_manager = self.unwrapped.reward_manager
        ignored = set(ignored_terms)
        active: list[tuple[int, str, float]] = []
        for index, (name, cfg) in enumerate(
            zip(
                self.reward_manager._term_names,
                self.reward_manager._term_cfgs,
                strict=True,
            )
        ):
            weight = float(cfg.weight)
            if name not in ignored and weight != 0.0:
                active.append((index, str(name), weight))
        if not active:
            raise ValueError("Fixed reward requires at least one nonzero reward term.")
        self._term_indices = tuple(item[0] for item in active)
        self._weights = torch.tensor([item[2] for item in active])
        self.spec = FeatureSpec(
            dim=len(active), names=tuple(item[1] for item in active)
        )
        self.device = torch.device(
            device if device is not None else getattr(env, "device", "cpu")
        )
        self.capture_terminal_observations = bool(capture_terminal_observations)
        self._latest_features: torch.Tensor | None = None
        self._latest_environment_reward: torch.Tensor | None = None
        self._latest_terminal_observations: Observation | None = None
        self._original_compute: Callable[[float], torch.Tensor] | None = None

    @property
    def weights(self) -> torch.Tensor:
        return self._weights.clone()

    @property
    def step_dt(self) -> float:
        return float(self.unwrapped.step_dt)

    def install(self) -> None:
        if self._original_compute is not None:
            raise RuntimeError("IsaacPreResetTap is already installed.")
        original_compute = self.reward_manager.compute
        self._original_compute = original_compute

        def compute_and_capture(dt: float) -> torch.Tensor:
            environment_reward = original_compute(dt)
            weighted_terms = self.reward_manager._step_reward[:, self._term_indices]
            weights = self._weights.to(
                device=weighted_terms.device, dtype=weighted_terms.dtype
            )
            raw_features = weighted_terms / weights.unsqueeze(0)
            if not torch.isfinite(raw_features).all():
                bad = (
                    (~torch.isfinite(raw_features)).nonzero(as_tuple=False)[:8].tolist()
                )
                raise ValueError(
                    f"Non-finite pre-reset reward features at indices {bad}."
                )
            self._latest_features = raw_features.detach().to(self.device).clone()
            self._latest_environment_reward = (
                environment_reward.detach().to(self.device).clone()
            )
            self._latest_terminal_observations = self._compute_terminal_observations()
            return environment_reward

        self.reward_manager.compute = compute_and_capture

    def uninstall(self) -> None:
        if self._original_compute is None:
            return
        self.reward_manager.compute = self._original_compute
        self._original_compute = None

    def _compute_terminal_observations(self) -> Observation | None:
        """Compute pre-reset observations, but only when an episode timed out.

        Terminated episodes are never bootstrapped, so the extra observation
        pass is worth its cost only when the environment truncates.
        """
        if not self.capture_terminal_observations:
            return None
        if not bool(self.unwrapped.termination_manager.time_outs.any()):
            return None
        return detached_observation(
            self.unwrapped.observation_manager.compute(), self.device
        )

    def terminal_observations(self) -> Observation | None:
        """Pre-reset observations of the latest step, or None when unavailable."""
        return self._latest_terminal_observations

    def features(self, env: Any | None = None) -> torch.Tensor:
        """Return the latest pre-reset raw feature matrix [N,D]."""
        del env
        if self._latest_features is None:
            raise RuntimeError(
                "No reward features have been captured; call env.step() first."
            )
        return self._latest_features

    def environment_reward(self) -> torch.Tensor:
        """Return the scalar reward [N] captured with the latest features."""
        if self._latest_environment_reward is None:
            raise RuntimeError(
                "No environment reward has been captured; call env.step() first."
            )
        return self._latest_environment_reward
