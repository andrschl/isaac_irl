"""Capture of everything Isaac computes before its automatic environment reset.

``ManagerBasedRLEnv.step`` computes terminations, then rewards, then resets the
terminated environments and only afterwards computes the observations it
returns. Reward terms and the true terminal observation are therefore only
reachable from inside the reward-manager call, which this tap wraps.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

import torch

from interfaces import FeatureSpec
from reward_features.manager_based import as_term_vector
from storage.observations import Observation, detached_observation


@dataclass(frozen=True, slots=True)
class TapTerm:
    """One captured reward term.

    ``step_reward_index`` points into the manager's per-step reward for terms
    the environment actually weights; it is None for feature-only terms, whose
    value must be computed directly.
    """

    name: str
    weight: float
    step_reward_index: int | None
    cfg: Any


def _select_terms(
    reward_manager: Any,
    *,
    ignored_terms: set[str],
    force_include_terms: set[str],
) -> tuple[TapTerm, ...]:
    """Choose captured terms in reward-manager order.

    The rule matches ``ManagerBasedFeatureSource``: skip ignored terms, skip
    zero-weight terms unless they are explicitly requested as features. Both
    feature sources must agree, or recorded demonstrations and online features
    would describe different reward spaces.
    """
    terms: list[TapTerm] = []
    for index, (name, cfg) in enumerate(
        zip(reward_manager._term_names, reward_manager._term_cfgs, strict=True)
    ):
        if name in ignored_terms:
            continue
        weight = float(cfg.weight)
        if weight == 0.0 and name not in force_include_terms:
            continue
        terms.append(
            TapTerm(
                name=str(name),
                weight=weight,
                step_reward_index=None if weight == 0.0 else index,
                cfg=cfg,
            )
        )
    if not terms:
        raise ValueError(
            "The reward manager exposes no captured terms: every term is either "
            "ignored or has weight zero without being requested as a feature."
        )
    return tuple(terms)


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
        self.unwrapped = getattr(env, "unwrapped", env)
        self.reward_manager = self.unwrapped.reward_manager
        self.terms = _select_terms(
            self.reward_manager,
            ignored_terms=set(ignored_terms),
            force_include_terms=set(force_include_terms),
        )
        self._weights = torch.tensor([term.weight for term in self.terms])
        self.spec = FeatureSpec(
            dim=len(self.terms), names=tuple(term.name for term in self.terms)
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
            raw_features = self._capture_features()
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

    def _capture_features(self) -> torch.Tensor:
        """Return raw unweighted term values [N,D] in reward-manager order.

        Weighted terms are recovered from the manager's per-step reward, which
        holds ``func(env) * weight``. Feature-only terms carry weight zero, so
        the manager skips them entirely and their value has to be computed here
        — still before the reset, like everything else this tap captures.
        """
        step_reward = self.reward_manager._step_reward
        num_envs = int(self.unwrapped.num_envs)
        columns: list[torch.Tensor] = []
        for term in self.terms:
            if term.step_reward_index is None:
                value = term.cfg.func(self.unwrapped, **(term.cfg.params or {}))
                columns.append(as_term_vector(value, num_envs).to(step_reward))
            else:
                columns.append(step_reward[:, term.step_reward_index] / term.weight)
        return torch.stack(columns, dim=1)

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
