"""Configuration for the retained off-policy SAC and PMD algorithms.

The public axes are deliberately orthogonal:

* algorithm chooses the policy update: reparameterized SAC or exponential
  policy mirror descent (PMD).
* value chooses the action-value representation: a scalar direct Q or a
  reward-independent successor-feature Q.

No configuration flag changes the type or meaning of an existing field.
Factories pattern-match these dataclasses and construct only the selected
components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias


@dataclass(slots=True)
class TransitionReplayCfg:
    """Dense transition replay limits and where the ring lives.

    learning_starts is measured in simulator transitions. capacity is also a
    transition count: replay is a ring buffer, so the oldest transitions are
    overwritten once it is full.

    storage_device chooses where those transitions sit. "auto" keeps them on
    the training device when it is a GPU, which removes a host-to-device copy
    from every critic update; "cpu" trades that for host memory when the ring
    would not fit.
    """

    learning_starts: int = 4_096
    capacity: int = 1_000_000
    storage_device: str = "auto"

    def __post_init__(self) -> None:
        if self.learning_starts <= 0:
            raise ValueError("replay.learning_starts must be > 0.")
        if self.capacity < self.learning_starts:
            raise ValueError("replay.capacity must be at least replay.learning_starts.")
        if self.storage_device not in {"auto", "cpu"}:
            raise ValueError(
                "replay.storage_device must be 'auto' or 'cpu', got "
                f"{self.storage_device!r}."
            )

    def resolve_storage_device(self, compute_device: str | object) -> str:
        """Return the device the ring should be allocated on."""
        if self.storage_device == "cpu":
            return "cpu"
        return "cpu" if "cpu" in str(compute_device) else str(compute_device)


@dataclass(slots=True)
class OffPolicyTrainCfg:
    """Data collection and Bellman-update hyperparameters shared by SAC/PMD.

    updates_per_transition is the update-to-data ratio (UTD): 1/16 means one
    critic optimizer step per sixteen newly collected simulator transitions.
    Increasing it spends more compute on the same data; it does not increase
    the simulator sample count.

    reset_interval_updates periodically reinitializes the learner - critics,
    actor, optimizers, entropy temperature - while keeping the replay. Networks
    trained at a high update-to-data ratio lose plasticity long before their
    data stops being informative, and relearning from the same replay is what
    recovers it. Zero disables resets.

    reset_scope chooses what a reset touches. "agent" reinitializes everything,
    which is the from-scratch recipe. "critic" leaves the actor and its
    optimizer alone, which is what fine-tuning an imitation policy needs: the
    value functions relearn, the behaviour prior survives.
    """

    gamma: float = 0.99
    rollout_steps: int = 8
    critic_batch_size: int = 256
    critic_learning_rate: float = 3.0e-4
    updates_per_transition: float = 1.0 / 16.0
    initial_critic_updates: int = 1_000
    target_polyak: float = 0.005
    max_grad_norm: float = 1.0
    reset_interval_updates: int = 0
    reset_scope: str = "agent"
    replay: TransitionReplayCfg = field(default_factory=TransitionReplayCfg)

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1].")
        for name, value in (
            ("rollout_steps", self.rollout_steps),
            ("critic_batch_size", self.critic_batch_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0.")
        if self.initial_critic_updates < 0:
            raise ValueError("initial_critic_updates must be >= 0.")
        if self.reset_interval_updates < 0:
            raise ValueError("reset_interval_updates must be >= 0.")
        if self.reset_scope not in {"agent", "critic"}:
            raise ValueError(
                f"reset_scope must be 'agent' or 'critic', got {self.reset_scope!r}."
            )
        for name, value in (
            ("critic_learning_rate", self.critic_learning_rate),
            ("updates_per_transition", self.updates_per_transition),
            ("max_grad_norm", self.max_grad_norm),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0.")
        if not 0.0 < self.target_polyak <= 1.0:
            raise ValueError("target_polyak must be in (0, 1].")

    def critic_updates_for(self, transition_count: int) -> int:
        """Return optimizer steps for transition_count new samples."""
        if transition_count <= 0:
            raise ValueError("transition_count must be > 0.")
        return max(1, round(self.updates_per_transition * transition_count))


@dataclass(slots=True)
class DirectQCfg:
    """Five-head scalar action-value ensemble used by direct SAC/PMD.

    Every head learns from a disjoint slice of one replay draw and bootstraps
    only from its paired EMA target. control_reduction decides what the actor
    optimizes: the ensemble mean, or the pessimistic minimum that keeps an
    over-estimating head from pulling the policy towards actions no head can
    justify.
    """

    type: ClassVar[str] = "direct"
    num_heads: int = 5
    width: int = 512
    residual_blocks: int = 2
    control_reduction: str = "mean"

    def __post_init__(self) -> None:
        _validate_value_network(
            self.num_heads, self.width, self.residual_blocks, name="direct Q"
        )
        _validate_reduction(self.control_reduction)


@dataclass(slots=True)
class SuccessorQCfg:
    """Bootstrapped vector successor-feature action-value ensemble.

    Successor TD targets contain features and dynamics only. Reward weights
    are applied when actions are scored, so changing a linear IRL reward never
    changes these Bellman targets. control_reduction applies to the scalarized
    per-head values only; feature occupancy is always the ensemble mean.

    scale_refit_ratio guards the output scale against features that are absent
    early: a feature only a competent policy ever produces looks like zero in
    the first rollouts. When the running feature scale outgrows the fitted one
    by this factor, the scale is refitted with predictions preserved.
    """

    type: ClassVar[str] = "successor"
    num_heads: int = 5
    control_reduction: str = "mean"
    width: int = 512
    residual_blocks: int = 2
    latent_dim: int = 128
    head_hidden_dim: int = 128
    return_scale_min: float = 1.0
    scale_refit_ratio: float = 2.0
    entropy_critic_learning_rate: float = 3.0e-4

    def __post_init__(self) -> None:
        _validate_value_network(
            self.num_heads, self.width, self.residual_blocks, name="successor Q"
        )
        _validate_reduction(self.control_reduction)
        for name, value in (
            ("latent_dim", self.latent_dim),
            ("head_hidden_dim", self.head_hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0.")
        if self.return_scale_min <= 0.0:
            raise ValueError("return_scale_min must be > 0.")
        if self.scale_refit_ratio <= 1.0:
            raise ValueError("scale_refit_ratio must be > 1.")
        if self.entropy_critic_learning_rate <= 0.0:
            raise ValueError("entropy_critic_learning_rate must be > 0.")


ValueBackendCfg: TypeAlias = DirectQCfg | SuccessorQCfg


@dataclass(slots=True)
class TanhGaussianPolicyCfg:
    """Shared bounded actor architecture and distribution limits.

    initial_std controls the random policy at startup. minimum_std prevents
    the learned policy from becoming numerically deterministic, and maximum_std
    keeps it out of the saturated regime: a weighted fit to a policy's own
    samples inflates variance while the critic is still uninformative, and a
    per-step trust region cannot stop that drift from compounding. Actions are
    produced by tanh and sent to the environment unchanged.
    """

    hidden_dims: tuple[int, ...] = (512, 256, 128)
    initial_std: float = 1.0
    minimum_std: float = 0.05
    maximum_std: float = 2.0
    action_bound: float = 1.0

    def __post_init__(self) -> None:
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("policy.hidden_dims must contain positive values.")
        if self.initial_std <= 0.0:
            raise ValueError("policy.initial_std must be > 0.")
        if not 0.0 < self.minimum_std <= self.initial_std <= self.maximum_std:
            raise ValueError(
                "Expected 0 < policy.minimum_std <= policy.initial_std <= "
                "policy.maximum_std."
            )
        if self.action_bound <= 0.0:
            raise ValueError("policy.action_bound must be > 0.")


@dataclass(slots=True)
class SacCfg:
    """Reparameterized maximum-entropy SAC policy update.

    The exact actor objective is

    E[entropy_temperature * log pi(a|s) - Q_soft(s,a)].

    Its unconstrained optimum is the beta=0 member of the exponential policy
    family: pi_target(a|s) is proportional to exp(Q/temperature). SAC
    intentionally owns no previous-policy snapshot.
    """

    type: ClassVar[str] = "sac"
    train: OffPolicyTrainCfg = field(default_factory=OffPolicyTrainCfg)
    policy: TanhGaussianPolicyCfg = field(default_factory=TanhGaussianPolicyCfg)
    value: ValueBackendCfg = field(default_factory=DirectQCfg)
    actor_learning_rate: float = 3.0e-4
    actor_batch_size: int = 256
    initial_entropy_temperature: float = 0.006
    automatic_entropy_tuning: bool = True
    entropy_learning_rate: float = 3.0e-4

    def __post_init__(self) -> None:
        _validate_actor(self.actor_learning_rate, self.actor_batch_size)
        if self.initial_entropy_temperature < 0.0:
            raise ValueError("initial_entropy_temperature must be >= 0.")
        if self.automatic_entropy_tuning and self.initial_entropy_temperature <= 0.0:
            raise ValueError(
                "Automatic entropy tuning requires a positive initial temperature."
            )
        if self.entropy_learning_rate <= 0.0:
            raise ValueError("entropy_learning_rate must be > 0.")


@dataclass(slots=True)
class PmdCfg:
    """Two-KL exponential policy mirror-descent update.

    The nonparametric target is

    pi_target(a|s) proportional to pi_old(a|s)^beta * exp(c * Q(s,a)).

    Candidates are sampled from pi_old, hence their importance log-weight is
    (beta - 1) * log_pi_old + c * Q. c is solved by bisection so the candidate
    target reaches candidate_target_kl. The neural actor is then fitted only
    up to actor_projection_kl. These separate constraints follow the practical
    MPO/REPS pattern and prevent reward scale from silently changing the step.

    critic_updates_per_projection is the policy-improvement cadence: the number
    of critic steps that evaluate the frozen pi_k before it is replaced. It is
    the actor-side analogue of OffPolicyTrainCfg.updates_per_transition, and it
    decides how much trust-region budget PMD spends per simulator transition.

    actor_spread_projection_kl bounds the part of the step that only widens or
    narrows the action distribution. Without it a projection can spend its whole
    trust region on variance and drift towards saturated, bang-bang actions.
    """

    type: ClassVar[str] = "pmd"
    entropy_temperature: float = 0.0
    train: OffPolicyTrainCfg = field(default_factory=OffPolicyTrainCfg)
    policy: TanhGaussianPolicyCfg = field(default_factory=TanhGaussianPolicyCfg)
    value: ValueBackendCfg = field(default_factory=SuccessorQCfg)
    beta: float = 1.0
    candidate_actions: int = 32
    candidate_target_kl: float = 0.05
    actor_projection_kl: float = 0.02
    actor_spread_projection_kl: float = 0.002
    q_coefficient_min: float = 1.0e-6
    q_coefficient_max: float = 1.0e3
    coefficient_bisection_steps: int = 32
    actor_learning_rate: float = 3.0e-4
    actor_batch_size: int = 512
    projection_steps: int = 8
    projection_backtrack_steps: int = 8
    critic_updates_per_projection: int = 8

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1].")
        if self.entropy_temperature < 0.0:
            raise ValueError("entropy_temperature must be >= 0.")
        for name, value in (
            ("candidate_actions", self.candidate_actions),
            ("coefficient_bisection_steps", self.coefficient_bisection_steps),
            ("projection_steps", self.projection_steps),
            ("projection_backtrack_steps", self.projection_backtrack_steps),
            ("critic_updates_per_projection", self.critic_updates_per_projection),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0.")
        if self.candidate_target_kl <= 0.0:
            raise ValueError("candidate_target_kl must be > 0.")
        if not 0.0 < self.actor_projection_kl < self.candidate_target_kl:
            raise ValueError(
                "actor_projection_kl must be positive and below candidate_target_kl."
            )
        if not 0.0 < self.actor_spread_projection_kl <= self.actor_projection_kl:
            raise ValueError(
                "actor_spread_projection_kl must be positive and at most "
                "actor_projection_kl."
            )
        if not 0.0 < self.q_coefficient_min < self.q_coefficient_max:
            raise ValueError("Expected 0 < q_coefficient_min < q_coefficient_max.")
        _validate_actor(self.actor_learning_rate, self.actor_batch_size)


OffPolicyAlgorithmCfg: TypeAlias = SacCfg | PmdCfg


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"mean", "min"}:
        raise ValueError(
            f"control_reduction must be 'mean' or 'min', got {reduction!r}."
        )


def _validate_value_network(
    num_heads: int, width: int, residual_blocks: int, *, name: str
) -> None:
    if num_heads <= 0:
        raise ValueError(f"{name} num_heads must be > 0.")
    if width <= 0:
        raise ValueError(f"{name} width must be > 0.")
    if residual_blocks <= 0:
        raise ValueError(f"{name} residual_blocks must be > 0.")


def _validate_actor(learning_rate: float, batch_size: int) -> None:
    if learning_rate <= 0.0:
        raise ValueError("actor_learning_rate must be > 0.")
    if batch_size <= 0:
        raise ValueError("actor_batch_size must be > 0.")
