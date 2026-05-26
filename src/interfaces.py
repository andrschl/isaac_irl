"""Shared protocols and data types for the runner / RL / reward-learner boundary.

Lives at the top level of `src/` so it can be imported by both `runner.*` and
`algorithms.*` without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch

from storage.feature_storage import FeatureBufCfg
from utils.runtime_context import RuntimeContext


class PolicyMode(str, Enum):
    """Pure (side-effect-free) policy action modes.

    Side-effectful training collection goes through ``RlAlgorithm.collect_action``
    instead and is not represented here.
    """

    TRAIN = "train"  # stochastic sampling
    INFERENCE = "inference"  # deterministic


@dataclass(slots=True)
class EnvTransition:
    """One vectorized env transition consumed by an RL algorithm.

    Shapes:
        actions:     [N, A]
        env_rewards: [N]
        rewards:     [N]
        dones:       [N]

    ``rewards`` is the reward the algorithm should learn from. ``env_rewards``
    is the raw environment reward kept for diagnostics/debugging.
    """

    obs: Any
    actions: torch.Tensor
    env_rewards: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_obs: Any
    extras: dict[str, Any]


@dataclass(slots=True)
class FeatureStep:
    """One vectorized feature step consumed by a reward learner.

    Shapes:
        features: [N, D]
        dones:    [N]
    """

    features: torch.Tensor
    dones: torch.Tensor


class RlAlgorithm(Protocol):
    """RL algorithm adapter owned by the runner.

    The action API is split:
      - ``collect_action`` runs during training collection. Side effects on the
        algorithm's internal storage (e.g. PPO's pending transition) are allowed.
      - ``act`` is the pure path used for imitator rollouts, validation, and
        final-video rollouts. It MUST NOT write to internal storage.
    """

    collects_rollouts: bool

    def collect_action(self, obs: Any) -> torch.Tensor:
        """Return actions [N, A] during training collection."""

    def act(self, obs: Any, *, mode: PolicyMode) -> torch.Tensor:
        """Return actions [N, A] without side effects."""

    def observe(self, transition: EnvTransition) -> None:
        """Store/process one environment transition."""

    def end_rollout(self, last_obs: Any) -> None:
        """Finalize a just-collected rollout. No-op for off-policy algorithms."""

    def update(self) -> dict[str, float]:
        """Run one algorithm update and return scalar metrics."""

    def train_metrics(self) -> dict[str, float]:
        """Cheap per-iteration metrics."""

    def eval_metrics(self) -> dict[str, float]:
        """Periodic eval metrics that may be more expensive."""

    def train_mode(self) -> None:
        """Put policy modules in training mode."""

    def eval_mode(self) -> None:
        """Put policy modules in eval mode."""

    def save_state(self) -> dict[str, Any]:
        """Return checkpoint payload for this algorithm."""

    def load_state(
        self, checkpoint: dict[str, Any], *, load_optimizer: bool = True
    ) -> None:
        """Load checkpoint payload for this algorithm."""


class IrlAlgorithm(Protocol):
    """Feature-based reward learner owned by the runner."""

    reward_model: torch.nn.Module
    gamma: float
    batch_size: int
    feature_names: list[str] | None
    expert_success_rate: float | None

    def init_expert_storage(
        self,
        runtime_ctx: RuntimeContext,
        *,
        cfg: FeatureBufCfg | None = None,
        num_envs: int = 1,
    ) -> None:
        """Initialize expert feature storage."""

    def init_imitator_storage(
        self,
        runtime_ctx: RuntimeContext,
        *,
        cfg: FeatureBufCfg | None = None,
    ) -> None:
        """Initialize imitator feature storage."""

    def observe(self, step: FeatureStep) -> None:
        """Store one vectorized feature step."""

    def clear_imitator(self) -> None:
        """Clear the current imitator feature buffer."""

    def finalize_imitator(self) -> None:
        """Flush in-progress imitator episodes."""

    def can_update(self) -> bool:
        """Whether expert and imitator buffers can produce update batches."""

    def update(self) -> dict[str, float]:
        """Run one reward update and return scalar metrics."""

    def per_feature_return_mean(
        self, feats: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-feature mean discounted returns. Shapes: feats [B, T, D], mask [B, T] -> [D]."""

    def sample_expert_feature_return_mean(
        self, device: torch.device | str
    ) -> torch.Tensor | None:
        """Sample expert episodes and return per-feature mean returns [D]."""

    def train_metrics(self) -> dict[str, float]:
        """Cheap per-iteration reward metrics."""

    def train_mode(self) -> None:
        """Put reward model in training mode."""

    def eval_mode(self) -> None:
        """Put reward model in eval mode."""

    def save_state(self) -> dict[str, Any]:
        """Return checkpoint payload for this reward learner."""

    def load_state(
        self, checkpoint: dict[str, Any], *, load_optimizer: bool = True
    ) -> None:
        """Load checkpoint payload for this reward learner."""
