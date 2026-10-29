"""Every experiment configuration dataclass, in one place.

Pure Python: no Isaac, no torch modules, no environment. Validation lives in
``__post_init__`` so a config is correct the moment it exists, wherever it was
built from - YAML, CLI, or a test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias

import torch

__all__ = [
    "ActorCriticCfg",
    "BCCfg",
    "DirectQCfg",
    "EnvCfg",
    "ExpertDataCfg",
    "FeatureBufCfg",
    "FeatureMapCfg",
    "IRLCfg",
    "OffPolicyTrainCfg",
    "PmdCfg",
    "PpoAlgoCfg",
    "RewardCfg",
    "RewardModelCfg",
    "RunnerCfg",
    "ValueBackendCfg",
    "SuccessorQCfg",
    "TanhGaussianPolicyCfg",
    "TrainCfg",
    "TransitionReplayCfg",
]


# --------------------------------------------------------------- storage


@dataclass(slots=True)
class FeatureBufCfg:
    """Buffer-specific configuration (chosen by the experiment)."""

    capacity_steps: int = 1_000_000
    store_device: str = "auto"
    """Where episodes live: "auto" is the training device, "cpu" host memory.

    Imitator episodes are rewritten every cycle, so they stay on the training
    device; a long-lived expert buffer is read once per update and belongs on
    the host.
    """
    store_dtype: torch.dtype = torch.float32
    min_ep_len: int = 2
    sample_weighted_by_length: bool = False

    def __post_init__(self) -> None:
        if self.store_device not in {"auto", "cpu"}:
            raise ValueError(
                f"`store_device` must be 'auto' or 'cpu', got {self.store_device!r}."
            )
        if self.capacity_steps <= 0:
            raise ValueError(
                f"`capacity_steps` must be > 0, got {self.capacity_steps}."
            )
        if self.min_ep_len <= 0:
            raise ValueError(f"`min_ep_len` must be > 0, got {self.min_ep_len}.")

    def resolve_store_device(self, compute_device: str | torch.device) -> str:
        """Return the device episodes are stored on."""
        if self.store_device == "cpu":
            return "cpu"
        return str(compute_device)


# ---------------------------------------------------------- reward model


@dataclass(slots=True)
class RewardModelCfg:
    """
    Feature-based reward model config.

    The reward model consumes features directly (not observations/actions).

    Dense model: hidden_dims, activation.
    Linear model: linear_projection, linear_projection_radius.
    Both: regularization, regularization_strength.
    """

    hidden_dims: tuple[int, ...] = (256, 256, 256)
    is_linear: bool = False
    activation: str = "elu"

    # Regularization (both dense and linear)
    regularization: str = "none"  # "none" | "l2"
    regularization_strength: float = 0.0

    # Linear-only: projection onto norm balls (applied after optimizer.step)
    linear_projection: str = "none"  # "none" | "l2_ball"
    linear_projection_radius: float = 1.0


# ------------------------------------------------------ off-policy (PMD)


@dataclass(slots=True)
class TransitionReplayCfg:
    """Dense transition replay limits and where the ring lives.

    learning_starts is measured in retained training transitions. capacity is also a
    transition count: replay is a ring buffer, so the oldest transitions are
    overwritten once it is full.

    storage_device chooses where those transitions sit. "auto" keeps them on
    the training device when it is a GPU, which removes a host-to-device copy
    from every critic update; "cpu" trades that for host memory when the ring
    would not fit.

    demonstration_path seeds the replay from recorded demonstrations, which are
    kept out of the ring and never evicted, and demonstration_fraction is their
    share of every sampled batch.
    """

    learning_starts: int = 4_096
    capacity: int = 1_000_000
    storage_device: str = "auto"
    demonstration_path: str | None = None
    demonstration_fraction: float = 0.5
    demonstration_episodes: int | None = None

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
        if not 0.0 < self.demonstration_fraction < 1.0:
            raise ValueError("replay.demonstration_fraction must lie in (0,1).")
        if self.demonstration_episodes is not None and self.demonstration_episodes <= 0:
            raise ValueError("replay.demonstration_episodes must be > 0 when set.")

    def resolve_storage_device(self, compute_device: str | object) -> str:
        """Return the device the ring should be allocated on."""
        if self.storage_device == "cpu":
            return "cpu"
        return "cpu" if "cpu" in str(compute_device) else str(compute_device)


@dataclass(slots=True)
class OffPolicyTrainCfg:
    """Data collection and Bellman-update hyperparameters for PMD.

    Collect episodes_per_batch complete attempts under a frozen policy, then
    Choose exactly one critic budget: a fixed critic_updates_per_batch, or
    critic_updates_per_transition times the newly retained training data.
    Neither counts ensemble heads or minibatch elements as optimizer steps.

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
    episodes_per_batch: int = 32
    critic_batch_size: int = 256
    critic_learning_rate: float = 3.0e-4
    critic_updates_per_batch: int | None = 1_000
    critic_updates_per_transition: float | None = None
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
            ("episodes_per_batch", self.episodes_per_batch),
            ("critic_batch_size", self.critic_batch_size),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0.")
        if (self.critic_updates_per_batch is None) == (
            self.critic_updates_per_transition is None
        ):
            raise ValueError(
                "Set exactly one of critic_updates_per_batch and "
                "critic_updates_per_transition; set the other to null."
            )
        if self.critic_updates_per_batch is not None and (
            not isinstance(self.critic_updates_per_batch, int)
            or isinstance(self.critic_updates_per_batch, bool)
            or self.critic_updates_per_batch <= 0
        ):
            raise ValueError("critic_updates_per_batch must be a positive integer.")
        ratio = self.critic_updates_per_transition
        if ratio is not None and (not math.isfinite(ratio) or ratio <= 0):
            raise ValueError("critic_updates_per_transition must be finite and > 0.")
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
            ("max_grad_norm", self.max_grad_norm),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0.")
        if not 0.0 < self.target_polyak <= 1.0:
            raise ValueError("target_polyak must be in (0, 1].")

    @property
    def minimum_critic_updates_per_batch(self) -> int:
        """Budget guaranteed even when every collected episode lasts one step."""
        if self.critic_updates_per_batch is not None:
            return self.critic_updates_per_batch
        return math.floor(self.episodes_per_batch * self.critic_updates_per_transition)


@dataclass(slots=True)
class DirectQCfg:
    """Scalar action-value ensemble used by the direct value backend.

    Every head learns from a disjoint slice of one replay draw.
    target_reduction decides what it bootstraps from. "independent" leaves each
    head trusting its own EMA target, which is M separate Q-learners each
    carrying a single critic's maximization bias; the actor then maximizes a
    quantity every head over-estimates in the same direction. "min" takes the
    minimum over target_subset_size heads drawn afresh each update -- clipped
    double-Q generalized to M heads, and at subset size two the classic rule.

    control_reduction is the same choice on the control side: what the actor
    optimizes, the ensemble mean or the pessimistic minimum.
    """

    type: ClassVar[str] = "direct"
    num_heads: int = 5
    width: int = 512
    residual_blocks: int = 2
    control_reduction: str = "min"
    target_reduction: str = "min"
    target_subset_size: int = 2

    def __post_init__(self) -> None:
        _validate_value_network(
            self.num_heads, self.width, self.residual_blocks, name="direct Q"
        )
        _validate_reduction(self.control_reduction)
        if self.target_reduction not in {"min", "independent"}:
            raise ValueError(
                "direct Q target_reduction must be 'min' or 'independent', got "
                f"{self.target_reduction!r}."
            )
        if self.target_reduction == "min" and not (
            2 <= self.target_subset_size <= self.num_heads
        ):
            raise ValueError(
                "direct Q target_subset_size must lie in "
                f"[2,{self.num_heads}], got {self.target_subset_size}; "
                "a clipped target needs at least two heads to minimize over, "
                "so set target_reduction='independent' to run without one."
            )


@dataclass(slots=True)
class SuccessorQCfg:
    """Bootstrapped vector successor-feature action-value ensemble.

    Successor TD targets contain features and dynamics only. Reward weights
    are applied when actions are scored, so changing a linear IRL reward never
    changes these Bellman targets. That is also why this ensemble has no
    in-target minimum: picking the pessimistic head needs a scalar, a scalar
    needs the reward weights, and a target chosen under w is no longer
    reward-independent. Pessimism therefore enters where scalarization already
    does: control_quantile selects a quantile of the scalarized per-head
    values, while feature occupancy itself remains the ensemble mean. 0.0 is the
    minimum, 0.25 the lower quartile, 0.5 the median, and 1.0 the maximum.
    Above 0.5 the score is optimistic: the actor is pulled toward actions the
    ensemble disagrees about, which is exploration rather than caution. That is
    only as informative as the disagreement itself, and these heads share a
    trunk, so their spread understates how little the ensemble knows.

    scale_refit_ratio guards the output scale against features that are absent
    early: a feature only a competent policy ever produces looks like zero in
    the first rollouts. When the running feature scale outgrows the fitted one
    by this factor, the scale is refitted with predictions preserved.
    """

    type: ClassVar[str] = "successor"
    num_heads: int = 5
    control_quantile: float = 0.25
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
        if not 0.0 <= self.control_quantile <= 1.0:
            raise ValueError(
                "successor Q control_quantile must be in [0.0, 1.0], got "
                f"{self.control_quantile}."
            )
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
    std_parameterization: str = "clamped_log"  # clamped_log | smooth_bounded

    def __post_init__(self) -> None:
        if self.std_parameterization not in {"clamped_log", "smooth_bounded"}:
            raise ValueError("Unknown policy.std_parameterization.")
        if self.std_parameterization == "smooth_bounded" and not (
            0 < self.minimum_std < self.initial_std < self.maximum_std < math.inf
        ):
            raise ValueError(
                "Smooth std requires finite 0 < minimum_std < initial_std < maximum_std."
            )
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
class PmdCfg:
    """Two-KL exponential policy mirror-descent update.

    The nonparametric target is

    pi_target(a|s) proportional to pi_old(a|s)^beta * exp(c * Q(s,a)).

    Candidates are sampled from pi_old, hence their importance log-weight is
    (beta - 1) * log_pi_old + c * Q. c is solved by bisection so the candidate
    target reaches candidate_target_kl. The neural actor is then fitted only
    up to actor_projection_kl. These separate constraints follow the practical
    MPO/REPS pattern and prevent reward scale from silently changing the step.

    policy_improvements_per_batch is independent of the critic compute budget.
    One improvement fits a single frozen projection target with up to
    projection_steps optimizer steps; it is not one improvement per TD step.

    actor_spread_projection_kl bounds the part of the step that only widens or
    narrows the action distribution. Without it a projection can spend its whole
    trust region on variance and drift towards saturated, bang-bang actions.
    """

    type: ClassVar[str] = "pmd"
    entropy_rate: float = 0.0
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
    policy_improvements_per_batch: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be in [0, 1].")
        if self.entropy_rate < 0.0:
            raise ValueError("entropy_rate must be >= 0.")
        for name, value in (
            ("candidate_actions", self.candidate_actions),
            ("coefficient_bisection_steps", self.coefficient_bisection_steps),
            ("projection_steps", self.projection_steps),
            ("projection_backtrack_steps", self.projection_backtrack_steps),
            ("policy_improvements_per_batch", self.policy_improvements_per_batch),
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
        if (
            self.policy_improvements_per_batch
            > self.train.minimum_critic_updates_per_batch
        ):
            raise ValueError(
                "policy_improvements_per_batch cannot exceed the minimum critic "
                "budget per batch; increase the critic budget or episodes_per_batch."
            )


OffPolicyAlgorithmCfg: TypeAlias = PmdCfg


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


# ----------------------------------------------------------------- runner


@dataclass(slots=True)
class RunnerCfg:
    """Training-loop configuration: cadence, validation, and buffers."""

    steps_per_env_per_cycle: int = 24
    save_interval: int = 50
    rl_updates_per_cycle: int = 1
    reward_updates_per_cycle: int = 1

    # Where the reward learner's imitator episodes come from. "collected"
    # reuses the features the policy's own collection already produced;
    # "rollout" pays for a dedicated rollout (PPO only). PMD already supplies
    # complete episodes and always reuses them via "collected".
    imitator_source: str = "collected"
    imitator_rollout_steps_per_env: int | None = None
    imitator_buffer: FeatureBufCfg = field(
        default_factory=lambda: FeatureBufCfg(min_ep_len=1)
    )
    expert_buffer: FeatureBufCfg = field(
        default_factory=lambda: FeatureBufCfg(min_ep_len=1, store_device="cpu")
    )

    # Validate every N learning iterations; 0 disables. PMD evaluates complete
    # episodes; PPO's legacy inline evaluation uses a step window (None = 5 cycles).
    validation_interval: int = 50
    validation_steps_per_env: int | None = None
    validation_episodes: int = 10
    """Complete episodes per policy mode for PMD's boundary-only evaluation."""
    validation_interval_transitions: int | None = None
    """Validate every N retained training transitions instead of every N iterations.

    Two algorithms with different parallelism collect a different number of
    transitions per iteration, so an iteration-based cadence evaluates them at
    different points on the only axis they can be compared on. Set this when
    the runs are meant to be read against each other.
    """

    # Iterations always apply. The legacy training-only threshold can overshoot
    # by a collection batch; use total_transition_budget for a hard cap.
    transition_budget: int | None = None
    total_transition_budget: int | None = None
    """All environment steps, including evaluation and discarded vector lanes.

    Reserve a whole collection/evaluation batch before starting it. The run
    stops short of this limit when the next complete batch cannot fit.
    """

    success_rate_thresholds: tuple[float, ...] = ()
    return_per_step_thresholds: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        positive = {
            "steps_per_env_per_cycle": self.steps_per_env_per_cycle,
            "save_interval": self.save_interval,
            "rl_updates_per_cycle": self.rl_updates_per_cycle,
            "reward_updates_per_cycle": self.reward_updates_per_cycle,
            "validation_episodes": self.validation_episodes,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"`runner.{name}` must be > 0, got {value}.")
        if self.imitator_source not in {"collected", "rollout"}:
            raise ValueError(
                "`runner.imitator_source` must be 'collected' or 'rollout', got "
                f"{self.imitator_source!r}."
            )
        if (
            self.imitator_rollout_steps_per_env is not None
            and self.imitator_rollout_steps_per_env <= 0
        ):
            raise ValueError(
                "`runner.imitator_rollout_steps_per_env` must be > 0 or null, got "
                f"{self.imitator_rollout_steps_per_env}."
            )
        if self.validation_interval < 0:
            raise ValueError(
                f"`runner.validation_interval` must be >= 0, got {self.validation_interval}."
            )
        if (
            self.validation_steps_per_env is not None
            and self.validation_steps_per_env < 0
        ):
            raise ValueError(
                "`runner.validation_steps_per_env` must be >= 0 or null, got "
                f"{self.validation_steps_per_env}."
            )
        if (
            self.validation_interval_transitions is not None
            and self.validation_interval_transitions <= 0
        ):
            raise ValueError(
                "`runner.validation_interval_transitions` must be > 0 or null, got "
                f"{self.validation_interval_transitions}."
            )
        if self.transition_budget is not None and self.transition_budget <= 0:
            raise ValueError(
                f"`runner.transition_budget` must be > 0 or null, got {self.transition_budget}."
            )
        if (
            self.total_transition_budget is not None
            and self.total_transition_budget <= 0
        ):
            raise ValueError("runner.total_transition_budget must be > 0 or null.")


# ------------------------------------------------------------- experiment


@dataclass(slots=True)
class EnvCfg:
    """The Isaac task and how it is instantiated."""

    name: str | None = None
    device: str = "cuda:0"
    num_envs: int | None = None
    variant: str = "standard"
    """Task geometry preset; lift_easy changes only Franka Lift resets/goals."""
    lift_cube_xy_half_range: float = 0.02
    """Uniform cube XY reset half-width in metres for the lift_easy preset.

    Larger values broaden the initial-state distribution without changing
    goals, rewards, robot reset, or termination. The default preserves Easy Lift.
    """
    reward_curriculum: bool = False
    """Keep the task's reward-weight curriculum (environment-reward PPO only)."""

    is_finite_horizon: bool = False
    """The episode genuinely ends at the horizon, so nothing bootstraps past it.

    Left off by default: a continuing task cut by a time limit still has a real
    future beyond it, and dropping that bootstrap would bias every existing
    configuration. Only a task defined on a fixed horizon sets this.
    """

    terminal_success_reward: bool = False
    """Replace the task's dense reward with one unit for a goal held to the end.

    Every dense term stays in the reward manager at weight zero, so it remains
    a reward-learning feature; the environment simply stops paying for it.
    """

    success_hold_steps: int = 25
    """Consecutive final steps the goal must be held for terminal success."""

    success_threshold: float = 0.08
    """Object-to-goal distance (m) below which a Lift step counts as success."""

    add_success_bonus: bool = True
    """Inject a binary `success_bonus` reward term before the env is built.

    It is exposed as an IRL feature; `success_bonus_weight` decides whether it
    also contributes to the environment's own reward.
    """

    success_bonus_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.success_hold_steps <= 0:
            raise ValueError(
                f"`env.success_hold_steps` must be > 0, got {self.success_hold_steps}."
            )
        if self.terminal_success_reward:
            conflicts = {
                "is_finite_horizon must be true": not self.is_finite_horizon,
                "reward_curriculum must be false": self.reward_curriculum,
                "add_success_bonus must be false": self.add_success_bonus,
            }
            broken = sorted(name for name, is_broken in conflicts.items() if is_broken)
            if broken:
                raise ValueError(
                    "`env.terminal_success_reward` defines the task as a fixed "
                    f"horizon with one sparse reward, so {broken}."
                )
            if self.name not in {None, "Isaac-Lift-Cube-Franka-v0"}:
                raise ValueError(
                    "`env.terminal_success_reward` is only implemented for "
                    f"Isaac-Lift-Cube-Franka-v0, got {self.name!r}."
                )
        if self.variant not in {"standard", "lift_easy"}:
            raise ValueError("env.variant must be 'standard' or 'lift_easy'.")
        if not math.isfinite(self.lift_cube_xy_half_range) or not (
            0.0 < self.lift_cube_xy_half_range <= 0.1
        ):
            raise ValueError(
                "env.lift_cube_xy_half_range must be finite and in (0, 0.1] metres "
                "to keep cube x within the native Lift workspace [0.4, 0.6]."
            )
        if self.variant != "lift_easy" and self.lift_cube_xy_half_range != 0.02:
            raise ValueError(
                "env.lift_cube_xy_half_range requires env.variant='lift_easy'."
            )
        if self.variant == "lift_easy" and self.name not in {
            None,
            "Isaac-Lift-Cube-Franka-v0",
        }:
            raise ValueError(
                "env.variant='lift_easy' requires Isaac-Lift-Cube-Franka-v0."
            )
        if self.success_threshold <= 0.0:
            raise ValueError(
                f"`env.success_threshold` must be > 0, got {self.success_threshold}."
            )
        if self.num_envs is not None and self.num_envs <= 0:
            raise ValueError(f"`env.num_envs` must be > 0, got {self.num_envs}.")


@dataclass(slots=True)
class FeatureMapCfg:
    """Which environment reward terms become reward-learning features."""

    ignored_reward_terms: set[str] = field(default_factory=set)
    force_include_terms: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ExpertDataCfg:
    """Where demonstrations come from, and how many of them to use.

    Shared by the reward learner and behavioral cloning: both read the same
    file, and an ablation that subsets it must subset it identically for both.
    """

    path: str = ""
    num_trajectories: int | None = None
    subset_strategy: str = "first"  # "first" | "random"

    def __post_init__(self) -> None:
        if self.num_trajectories is not None and self.num_trajectories <= 0:
            raise ValueError(
                "`expert_data.num_trajectories` must be > 0 or null, got "
                f"{self.num_trajectories}."
            )
        if self.subset_strategy not in {"first", "random"}:
            raise ValueError(
                "`expert_data.subset_strategy` must be 'first' or 'random', got "
                f"{self.subset_strategy!r}."
            )


@dataclass(slots=True)
class IRLCfg:
    """How the reward model is fit to the expert/imitator return gap."""

    batch_size: int = 256
    gradient_steps: int = 1
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    reward_loss_coef: float = 1.0
    reward_learning_rate: float = 1e-4
    solver: str = "gradient"  # "gradient" | "exact_linear"
    discount_gamma: float | None = None
    """Discount for IRL returns; null inherits the policy algorithm's gamma."""
    normalize_returns_by_episode_length: bool = True

    def __post_init__(self) -> None:
        if self.solver not in {"gradient", "exact_linear"}:
            raise ValueError(
                f"`irl.solver` must be gradient or exact_linear, got {self.solver!r}."
            )
        positive = {
            "batch_size": self.batch_size,
            "max_grad_norm": self.max_grad_norm,
            "reward_learning_rate": self.reward_learning_rate,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"`irl.{name}` must be > 0, got {value}.")
        if self.gradient_steps < 0 or (
            self.solver == "gradient" and self.gradient_steps == 0
        ):
            raise ValueError(
                "`irl.gradient_steps` must be > 0 for gradient and >= 0 for exact_linear."
            )
        if self.solver == "exact_linear" and (
            self.gradient_steps != 0 or self.weight_decay != 0
        ):
            raise ValueError(
                "`irl.solver=exact_linear` requires gradient_steps=0 and weight_decay=0."
            )
        if self.weight_decay < 0.0:
            raise ValueError(
                f"`irl.weight_decay` must be >= 0, got {self.weight_decay}."
            )
        if self.reward_loss_coef < 0.0:
            raise ValueError(
                f"`irl.reward_loss_coef` must be >= 0, got {self.reward_loss_coef}."
            )
        if self.discount_gamma is not None and not 0.0 < self.discount_gamma <= 1.0:
            raise ValueError(
                f"`irl.discount_gamma` must be in (0, 1], got {self.discount_gamma}."
            )


@dataclass(slots=True)
class BCCfg:
    """Behavioral cloning as a regularizer on the policy loss.

    ``L_actor = (1 - weight) * L_RL + weight * L_BC``. At ``weight=0`` no
    loader, batch, or gradient is constructed; at ``1`` the update is BC only.
    The actor optimizer is owned by the policy algorithm, so no learning rate
    appears here. ``val_fraction`` holds out whole expert episodes whose BC
    loss is reported but never trained on.
    """

    weight: float = 0.0
    loss_type: str = "nll"  # "nll" | "mse"
    batch_size: int = 256
    val_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"`bc.weight` must be in [0, 1], got {self.weight}.")
        if self.loss_type not in {"nll", "mse"}:
            raise ValueError(
                f"`bc.loss_type` must be 'nll' or 'mse', got {self.loss_type!r}."
            )
        if self.batch_size <= 0:
            raise ValueError(f"`bc.batch_size` must be > 0, got {self.batch_size}.")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(
                f"`bc.val_fraction` must be in [0, 1), got {self.val_fraction}."
            )


@dataclass(slots=True)
class RewardCfg:
    """What the agent maximizes, and the model that represents it.

    ``source`` is the whole decision: ``learned`` fits the reward model to
    demonstrations and hands the policy its output, ``environment`` hands the
    policy the environment's own reward. ``fit_diagnostic`` keeps the reward
    learner running alongside an environment-reward run, which is how a learned
    reward is compared against the one it is trying to recover.
    """

    source: str = "learned"
    fit_diagnostic: bool = False
    # Only the additive source uses these; defaults preserve the original sum.
    environment_weight: float = 1.0
    learned_weight: float = 1.0
    learned_feature_scales: dict[str, float] | None = None
    model: RewardModelCfg = field(default_factory=RewardModelCfg)

    def __post_init__(self) -> None:
        coefficients = (self.environment_weight, self.learned_weight)
        if any(not math.isfinite(value) or value < 0 for value in coefficients):
            raise ValueError("Reward mixture weights must be finite and nonnegative.")
        if sum(coefficients) == 0:
            raise ValueError("At least one reward mixture weight must be positive.")
        if self.source != "environment_plus_learned" and coefficients != (1.0, 1.0):
            raise ValueError(
                "Reward mixture weights require source=environment_plus_learned; "
                "single-source rewards have unit weight."
            )
        if self.source not in {"learned", "environment", "environment_plus_learned"}:
            raise ValueError(
                f"Unknown reward.source {self.source!r}; use learned, environment, or environment_plus_learned."
            )
        if self.fit_diagnostic and self.source != "environment":
            raise ValueError(
                "`reward.fit_diagnostic` only applies to `reward.source: environment`; "
                "a learned reward is always fit."
            )
        if self.learned_feature_scales is not None:
            scales = self.learned_feature_scales
            if (
                not isinstance(scales, dict)
                or not self.learns_a_reward
                or not self.model.is_linear
                or not scales
            ):
                raise ValueError(
                    "`reward.learned_feature_scales` requires a nonempty linear learned reward."
                )
            for name, scale in scales.items():
                if (
                    not isinstance(name, str)
                    or not name
                    or not isinstance(scale, (int, float))
                    or not math.isfinite(scale)
                    or scale <= 0
                ):
                    raise ValueError(
                        f"Invalid learned feature scale {name!r}: {scale!r}; "
                        "names must be nonempty and scales finite positive numbers."
                    )

    @property
    def learns_a_reward(self) -> bool:
        """Whether a reward learner is constructed and updated."""
        return self.source != "environment" or self.fit_diagnostic


@dataclass(slots=True)
class ActorCriticCfg:
    """rsl_rl actor/critic architecture used by the PPO baseline."""

    actor_hidden_dims: tuple[int, ...] = (256, 128, 64)
    critic_hidden_dims: tuple[int, ...] = (256, 128, 64)
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False


@dataclass(slots=True)
class PpoAlgoCfg:
    """PPO hyperparameters; ``entropy_rate`` is integrated using env ``dt``."""

    type: ClassVar[str] = "ppo"
    learning_rate: float = 1.0e-4
    gamma: float = 0.98
    lam: float = 0.95
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    entropy_rate: float = 0.006
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    normalize_advantage_per_mini_batch: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"`algo.gamma` must be in (0, 1], got {self.gamma}.")
        if self.learning_rate <= 0.0:
            raise ValueError(
                f"`algo.learning_rate` must be > 0, got {self.learning_rate}."
            )
        if self.entropy_rate < 0.0:
            raise ValueError(
                f"`algo.entropy_rate` must be >= 0, got {self.entropy_rate}."
            )


@dataclass(slots=True)
class TrainCfg:
    """One experiment, fully resolved."""

    experiment_name: str = "default_experiment"
    run_name: str = ""
    logger: str = "wandb"  # "wandb" | "noop"
    wandb_project: str = "isaac_irl"

    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    init_policy_checkpoint: str | None = None

    seed: int = 42
    max_iterations: int = 1500
    device: str = "cuda:0"

    env: EnvCfg = field(default_factory=EnvCfg)
    feature_map: FeatureMapCfg = field(default_factory=FeatureMapCfg)
    expert_data: ExpertDataCfg = field(default_factory=ExpertDataCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    irl: IRLCfg = field(default_factory=IRLCfg)
    bc: BCCfg = field(default_factory=BCCfg)
    runner: RunnerCfg = field(default_factory=RunnerCfg)
    actor_critic: ActorCriticCfg = field(default_factory=ActorCriticCfg)
    algorithm: PpoAlgoCfg | PmdCfg = field(default_factory=PpoAlgoCfg)

    def __post_init__(self) -> None:
        if self.logger not in {"wandb", "noop"}:
            raise ValueError(
                f"`logger` must be 'wandb' or 'noop', got {self.logger!r}."
            )
        if self.max_iterations <= 0:
            raise ValueError(
                f"`max_iterations` must be > 0, got {self.max_iterations}."
            )
        if self.resume and self.init_policy_checkpoint is not None:
            raise ValueError(
                "`--init_policy_checkpoint` is mutually exclusive with `--resume`."
            )
        if self.env.reward_curriculum and (
            not self.is_on_policy
            or self.reward.source != "environment"
            or self.reward.fit_diagnostic
        ):
            raise ValueError(
                "env.reward_curriculum=true requires PPO with reward.source=environment "
                "and reward.fit_diagnostic=false. PMD/IRL require fixed environment "
                "weights for reward-feature/replay consistency; set it to false."
            )
        self._validate_reward_basis()
        self._validate_demonstration_paths()
        if not self.is_on_policy:
            if self.runner.imitator_source != "collected":
                raise ValueError(
                    "Episode-batched PMD requires runner.imitator_source=collected; reuse its complete episodes."
                )
            if self.runner.rl_updates_per_cycle != 1:
                raise ValueError(
                    "PMD requires runner.rl_updates_per_cycle=1; use policy_improvements_per_batch."
                )

    def _validate_reward_basis(self) -> None:
        """Check that the reward the policy maximizes can actually be written.

        Fixed and learned coefficients share a named feature basis. Mixed
        sparse+learned reward fixes terminal success; pure learned reward may
        explicitly include it among the fitted coordinates.
        """
        if self.reward.source == "environment_plus_learned" and not (
            self.reward.model.is_linear
        ):
            raise ValueError(
                "`reward.source: environment_plus_learned` adds a fixed and a "
                "learned coefficient per feature, so it needs "
                "`reward.model.is_linear: true`."
            )
        if (
            self.reward.source == "environment_plus_learned"
            and self.reward.learned_feature_scales is not None
            and "terminal_success" in self.reward.learned_feature_scales
        ):
            raise ValueError(
                "Mixed sparse+learned reward fixes terminal_success at +1; "
                "do not also learn it."
            )
        if not self.env.terminal_success_reward:
            return
        if self.reward.learns_a_reward and not self.reward.model.is_linear:
            raise ValueError(
                "`env.terminal_success_reward` holds one reward coefficient "
                "fixed while the rest are learned, which needs "
                "`reward.model.is_linear: true`."
            )

    def _validate_demonstration_paths(self) -> None:
        """One demonstration file per run, read by everything that reads one.

        Behavioral cloning, the reward learner's expert buffer and the
        protected replay partition are three readers of the same episodes. Two
        paths that drift apart would have an arm score itself against a
        different expert than it imitates, and nothing in the run would say so.
        """
        if self.is_on_policy:
            return
        replay_path = self.algorithm.train.replay.demonstration_path
        if not replay_path or not self.expert_data.path:
            return
        if replay_path != self.expert_data.path:
            raise ValueError(
                "`algo.train.replay.demonstration_path` and `expert_data.path` "
                f"name different files ({replay_path!r} and "
                f"{self.expert_data.path!r}); a run has one demonstration set."
            )

    @property
    def is_on_policy(self) -> bool:
        return isinstance(self.algorithm, PpoAlgoCfg)

    @property
    def gamma(self) -> float:
        """The policy algorithm's discount, wherever it is configured."""
        return (
            self.algorithm.gamma
            if isinstance(self.algorithm, PpoAlgoCfg)
            else self.algorithm.train.gamma
        )
