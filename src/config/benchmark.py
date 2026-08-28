"""Task profiles for the fixed-reward sample-efficiency benchmark.

One profile fixes everything task-specific — environment id, environment count,
transition budget, evaluation protocol, task thresholds, and the tuned on-policy
baseline — so that PPO, SAC and PMD differ only in how they learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from config.off_policy import (
    DirectQCfg,
    OffPolicyTrainCfg,
    PmdCfg,
    SacCfg,
    SuccessorQCfg,
    TanhGaussianPolicyCfg,
)

AlgorithmName = Literal["ppo", "sac", "pmd"]


@dataclass(slots=True)
class PpoBaselineCfg:
    """On-policy baseline hyperparameters, one tuned set per task."""

    rollout_steps: int
    hidden_dims: tuple[int, ...]
    learning_rate: float
    gamma: float
    entropy_coef: float
    lam: float = 0.95
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    desired_kl: float = 0.01
    init_noise_std: float = 1.0
    max_grad_norm: float = 1.0
    value_loss_coef: float = 1.0

    def __post_init__(self) -> None:
        if self.rollout_steps <= 0:
            raise ValueError("ppo.rollout_steps must be > 0.")
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("ppo.hidden_dims must contain positive values.")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("entropy_coef", self.entropy_coef),
            ("desired_kl", self.desired_kl),
        ):
            if value <= 0.0:
                raise ValueError(f"ppo.{name} must be > 0.")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.lam <= 1.0:
            raise ValueError("ppo.gamma and ppo.lam must be in (0, 1].")


@dataclass(slots=True)
class BenchmarkTask:
    """Everything the benchmark needs to know about one task.

    ``action_clip`` is the environment's action bound and the tanh policies'
    bound at once: the off-policy actors can never reach it, and the on-policy
    baseline is clipped to it, so all three algorithms act in the same space.
    """

    name: str
    task_id: str
    num_envs: int
    transition_budget: int
    evaluation_interval_transitions: int
    evaluation_steps: int
    action_clip: float
    ppo: PpoBaselineCfg
    sac: SacCfg
    pmd: PmdCfg
    return_per_step_thresholds: tuple[float, ...] = ()
    success_rate_thresholds: tuple[float, ...] = ()
    success_threshold: float | None = None
    add_success_bonus: bool = False
    ignored_reward_terms: frozenset[str] = frozenset()
    force_include_reward_terms: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name, value in (
            ("num_envs", self.num_envs),
            ("transition_budget", self.transition_budget),
            ("evaluation_interval_transitions", self.evaluation_interval_transitions),
            ("evaluation_steps", self.evaluation_steps),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0.")
        if self.action_clip <= 0.0:
            raise ValueError("action_clip must be > 0.")
        if any(
            not 0.0 < threshold <= 1.0 for threshold in self.success_rate_thresholds
        ):
            raise ValueError("success_rate_thresholds must lie in (0, 1].")
        if self.success_rate_thresholds and self.success_threshold is None:
            raise ValueError(
                "success_rate_thresholds require a success_threshold to measure."
            )
        # The bounded actors are built at the task's action bound.
        self.sac = replace(
            self.sac, policy=replace(self.sac.policy, action_bound=self.action_clip)
        )
        self.pmd = replace(
            self.pmd, policy=replace(self.pmd.policy, action_bound=self.action_clip)
        )


def _sac_cfg(
    *,
    train: OffPolicyTrainCfg | None = None,
    policy: TanhGaussianPolicyCfg | None = None,
    value: DirectQCfg | SuccessorQCfg | None = None,
) -> SacCfg:
    """Maximum-entropy SAC with the scalar direct-Q ensemble."""
    return SacCfg(
        train=train or OffPolicyTrainCfg(),
        policy=policy or TanhGaussianPolicyCfg(),
        value=value or DirectQCfg(num_heads=5),
        initial_entropy_temperature=0.006,
        automatic_entropy_tuning=True,
    )


def _pmd_cfg(
    *,
    train: OffPolicyTrainCfg | None = None,
    policy: TanhGaussianPolicyCfg | None = None,
    value: DirectQCfg | SuccessorQCfg | None = None,
) -> PmdCfg:
    """Exponential PMD; the value backend is selected per run."""
    return PmdCfg(
        train=train or OffPolicyTrainCfg(),
        policy=policy or TanhGaussianPolicyCfg(),
        value=value or SuccessorQCfg(num_heads=5),
        beta=1.0,
        entropy_temperature=0.0,
    )


def cartpole_task() -> BenchmarkTask:
    """Classic cartpole: fast, dense, and solved within a few 10k transitions."""
    return BenchmarkTask(
        name="cartpole",
        task_id="Isaac-Cartpole-v0",
        num_envs=256,
        transition_budget=150_000,
        evaluation_interval_transitions=4_096,
        evaluation_steps=250,
        action_clip=1.0,
        # Break-even, "task solved", and near-optimal control.
        return_per_step_thresholds=(0.0, 3.0 / 250.0, 3.9 / 250.0),
        ignored_reward_terms=frozenset({"success_bonus"}),
        ppo=PpoBaselineCfg(
            rollout_steps=16,
            hidden_dims=(32, 32),
            learning_rate=1.0e-3,
            gamma=0.99,
            entropy_coef=0.005,
        ),
        sac=_sac_cfg(),
        pmd=_pmd_cfg(),
    )


def _lift_policy() -> TanhGaussianPolicyCfg:
    """Match the actor capacity of the task's tuned on-policy baseline."""
    return TanhGaussianPolicyCfg(hidden_dims=(256, 128, 64))


def _lift_off_policy_train() -> OffPolicyTrainCfg:
    """Lift trades update-to-data ratio for wall-clock.

    Manipulation runs need millions of transitions, and every critic step costs
    an ensemble of forward/backward passes, so the profile halves the Cartpole
    UTD. It is the same setting for SAC and PMD, so the comparison between them
    is unaffected; only their distance to the compute frontier is.
    """
    return OffPolicyTrainCfg(updates_per_transition=1.0 / 32.0)


def lift_task() -> BenchmarkTask:
    """Franka cube lift: sparse-ish manipulation with a real success metric.

    ``success_bonus`` is injected with weight zero, so it is a feature the
    successor critic already carries and the environment reward ignores — the
    same feature space the learned-reward stage will use.
    """
    return BenchmarkTask(
        name="lift",
        task_id="Isaac-Lift-Cube-Franka-v0",
        num_envs=512,
        transition_budget=3_000_000,
        # Above one full episode per environment (300 steps), so training sees
        # episodes end instead of being reset by every validation.
        evaluation_interval_transitions=204_800,
        evaluation_steps=250,
        action_clip=1.0,
        success_rate_thresholds=(0.25, 0.5, 0.9),
        success_threshold=0.08,
        add_success_bonus=True,
        force_include_reward_terms=frozenset({"success_bonus"}),
        ppo=PpoBaselineCfg(
            rollout_steps=24,
            hidden_dims=(256, 128, 64),
            learning_rate=1.0e-4,
            gamma=0.98,
            entropy_coef=0.006,
        ),
        sac=_sac_cfg(
            train=_lift_off_policy_train(),
            policy=_lift_policy(),
            value=DirectQCfg(num_heads=3, hidden_dims=(256, 128, 64)),
        ),
        pmd=_pmd_cfg(
            train=_lift_off_policy_train(),
            policy=_lift_policy(),
            value=SuccessorQCfg(
                num_heads=3,
                trunk_hidden_dims=(256, 128),
                latent_dim=64,
                head_hidden_dim=64,
                entropy_critic_hidden_dims=(256, 128, 64),
            ),
        ),
    )


BENCHMARK_TASKS = {"cartpole": cartpole_task, "lift": lift_task}


def build_task(name: str) -> BenchmarkTask:
    """Return a fresh profile for one benchmark task."""
    if name not in BENCHMARK_TASKS:
        raise ValueError(
            f"Unknown benchmark task {name!r}; expected one of "
            f"{sorted(BENCHMARK_TASKS)}."
        )
    return BENCHMARK_TASKS[name]()


@dataclass(slots=True)
class FixedRewardBenchmarkCfg:
    """One benchmark run: which algorithm, on which task profile."""

    algorithm: AlgorithmName
    task: BenchmarkTask = field(default_factory=cartpole_task)
    seed: int = 1
    device: str = "cuda:0"
    project: str = "isaac-irl-sample-efficiency"
    group: str = "fixed-reward-v4"
    run_name: str | None = None
    log_root: str = "logs/benchmark"
    record_video: bool = False
    video_length: int = 250
