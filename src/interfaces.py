"""Shared protocols and data types for the runner / RL / reward-learner boundary.

Lives at the top level of `src/` so it can be imported by both `runner.*` and
`algorithms.*` without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from storage.feature_storage import FeatureBufCfg
from utils.runtime_context import RuntimeContext


@dataclass(slots=True)
class FeatureStep:
    """One vectorized feature step consumed by a reward learner.

    Shapes:
        features: [N, D]
        dones:    [N]
    """

    features: torch.Tensor
    dones: torch.Tensor


@dataclass(frozen=True)
class FeatureSpec:
    """Static description of what a :class:`FeatureSource` emits.

    `dim` is the per-step feature dimensionality D. `names` is optional
    metadata used for logging; when present, its length must equal `dim`.
    """

    dim: int
    names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"FeatureSpec.dim must be > 0, got {self.dim}.")
        if self.names is not None and len(self.names) != self.dim:
            raise ValueError(
                f"FeatureSpec.names length ({len(self.names)}) must equal dim ({self.dim})."
            )


@dataclass(slots=True)
class StepContext:
    """Inputs available to a :class:`FeatureSource` at one vectorized step.

    Built by the rollout helper. Backends consume whichever fields they need:
    Isaac's reward-manager FeatureSource reads only `ctx.env`; tabular MDPs
    consume `(obs, action, next_obs)`. `info` carries any extra dict the env
    emitted on the last `step()` (e.g. rsl_rl's `time_outs`).
    """

    env: Any
    obs: Any
    action: torch.Tensor
    next_obs: Any
    env_reward: torch.Tensor
    done: torch.Tensor
    info: dict[str, Any]


class FeatureSource(Protocol):
    """Compute per-step features φ(...) for the visited state(s).

    Implementations declare what they emit via `spec` and produce a
    `[num_envs, spec.dim]` tensor from a `StepContext`. The protocol is the
    one per-env seam that genuinely varies across backends — everything
    downstream consumes [N, D] features uniformly.
    """

    spec: FeatureSpec

    def features(self, ctx: StepContext) -> torch.Tensor: ...


class Actor(Protocol):
    """Action-producing facade over an RL algorithm's policy.

    `act` samples stochastically (training behavior). `act_inference` returns
    the deterministic mean. The split exists so callers select the mode by
    method, not by an enum — Backends only need to expose what they support.
    """

    def act(self, obs: Any) -> torch.Tensor: ...
    def act_inference(self, obs: Any) -> torch.Tensor: ...


class VecEnv(Protocol):
    """Vectorized env contract mirrored from rsl_rl's `VecEnv`.

    Adapters that bridge to rsl_rl-shaped envs translate `num_actions` ->
    `action_dim` so the core uses one consistent name.
    """

    num_envs: int
    action_dim: int
    device: torch.device | str
    max_episode_length: int | torch.Tensor

    def get_observations(self) -> Any: ...

    def step(
        self, actions: torch.Tensor
    ) -> tuple[Any, torch.Tensor, torch.Tensor, dict]: ...


class RlAlgorithm(Protocol):
    """RL algorithm adapter owned by the runner.

    `actor` exposes the side-effect-free policy used by validation, imitator
    rollouts, and final-video rollouts. `update()` is responsible for any
    env collection the algorithm needs — the runner never steps the env for
    RL collection.
    """

    @property
    def actor(self) -> Actor:
        """Side-effect-free policy facade."""

    def update(self) -> dict[str, float]:
        """Collect (if needed) and run one algorithm update; return scalar metrics."""

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

    def load_state(self, checkpoint: dict[str, Any], *, load_optimizer: bool = True) -> None:
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

    def per_feature_return_mean(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-feature mean discounted returns. Shapes: feats [B, T, D], mask [B, T] -> [D]."""

    def sample_expert_feature_return_mean(self, device: torch.device | str) -> torch.Tensor | None:
        """Sample expert episodes and return per-feature mean returns [D]."""

    def train_metrics(self) -> dict[str, float]:
        """Cheap per-iteration reward metrics."""

    def train_mode(self) -> None:
        """Put reward model in training mode."""

    def eval_mode(self) -> None:
        """Put reward model in eval mode."""

    def save_state(self) -> dict[str, Any]:
        """Return checkpoint payload for this reward learner."""

    def load_state(self, checkpoint: dict[str, Any], *, load_optimizer: bool = True) -> None:
        """Load checkpoint payload for this reward learner."""
