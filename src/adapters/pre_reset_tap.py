"""Capture of everything Isaac computes before its automatic environment reset.

``ManagerBasedRLEnv.step`` computes terminations, then rewards, then resets the
terminated environments and only afterwards computes the observations it
returns. Reward terms and the true terminal observation are therefore only
reachable from inside the reward-manager call, which this tap wraps.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Callable

import torch

from interfaces import FeatureSpec
from reward.manager_terms import select_reward_terms
from storage.observations import Observation, detached_observation


class IsaacPreResetTap:
    """Capture raw reward terms, the env reward, and terminal observations."""

    def __init__(
        self,
        env: Any,
        *,
        ignored_terms: Iterable[str] = (),
        force_include_terms: Iterable[str] = (),
        device: str | torch.device | None = None,
        capture_terminal_observations: bool = False,
    ) -> None:
        self.env = env
        self.unwrapped = env.unwrapped
        self.reward_manager = self.unwrapped.reward_manager
        self.terms = select_reward_terms(
            self.reward_manager,
            ignored_terms=ignored_terms,
            force_include_terms=force_include_terms,
        )
        self.device = torch.device(device if device is not None else env.device)
        self._weights = torch.tensor(
            [term.weight for term in self.terms], device=self.device
        )
        self.spec = FeatureSpec(
            dim=len(self.terms), names=tuple(term.name for term in self.terms)
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
        """Wrap the reward manager so every reward computation is captured."""
        if self._original_compute is not None:
            raise RuntimeError("IsaacPreResetTap is already installed.")
        original_compute = self.reward_manager.compute
        self._original_compute = original_compute

        def compute_and_capture(dt: float) -> torch.Tensor:
            environment_reward = original_compute(dt)
            self._capture_step(environment_reward)
            return environment_reward

        self.reward_manager.compute = compute_and_capture

    def uninstall(self) -> None:
        """Restore the reward manager's own compute."""
        if self._original_compute is None:
            raise RuntimeError("IsaacPreResetTap is not installed.")
        self.reward_manager.compute = self._original_compute
        self._original_compute = None

    @contextmanager
    def installed(self) -> Iterator["IsaacPreResetTap"]:
        """Install for the duration of a block, and always restore afterwards."""
        self.install()
        try:
            yield self
        finally:
            self.uninstall()

    def _capture_step(self, environment_reward: torch.Tensor) -> None:
        """Record what this step produced, before Isaac resets anything."""
        raw_features = self._capture_features()
        if not torch.isfinite(raw_features).all():
            bad = (~torch.isfinite(raw_features)).nonzero(as_tuple=False)[:8].tolist()
            raise ValueError(f"Non-finite pre-reset reward features at indices {bad}.")
        self._latest_features = raw_features.detach().to(self.device).clone()
        self._latest_environment_reward = (
            environment_reward.detach().to(self.device).clone()
        )
        self._latest_terminal_observations = self._compute_terminal_observations()

    def _require_constant_weights(self) -> None:
        """Fail if the environment has changed a reward weight underneath us.

        Features are recovered by dividing the manager's weighted per-step
        reward by the weight snapshotted at construction, so a weight the
        environment changes later silently rescales that feature -- and cannot
        be caught downstream, because the same stale weight is used to score
        it, cancelling the error exactly. Isaac Lab's own manipulation tasks do
        this: Lift multiplies its action_rate and joint_vel penalties by 1000
        once common_step_counter passes 10,000 steps.
        """
        changed = {
            term.name: (term.weight, float(term.cfg.weight))
            for term in self.terms
            if float(term.cfg.weight) != term.weight
        }
        if changed:
            raise ValueError(
                "Reward weights changed during the run, so the ground-truth "
                f"reward is not fixed: {changed} (term: was, now). A reward "
                "curriculum is the usual cause; drop those curriculum terms "
                "from the environment configuration."
            )

    def _capture_features(self) -> torch.Tensor:
        """Return raw unweighted term values [N,D] in reward-manager order.

        Weighted terms are recovered from the manager's per-step reward, which
        holds ``func(env) * weight``. Feature-only terms carry weight zero, so
        the manager skips them entirely and their value has to be computed here
        — still before the reset, like everything else this tap captures.
        """
        self._require_constant_weights()
        step_reward = self.reward_manager._step_reward
        num_envs = int(self.unwrapped.num_envs)
        columns: list[torch.Tensor] = []
        for term in self.terms:
            if term.contributes_to_reward:
                # Recover the raw term by undoing the weight the manager applied.
                columns.append(step_reward[:, term.index] / term.weight)
            else:
                columns.append(term.evaluate(self.unwrapped, num_envs).to(step_reward))
        return torch.stack(columns, dim=1)

    def _compute_terminal_observations(self) -> Observation | None:
        """Capture true successor observations for both failures and timeouts."""
        if not self.capture_terminal_observations:
            return None
        endings = self.unwrapped.termination_manager
        if not bool((endings.time_outs | endings.terminated).any()):
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
