"""Configuration for the fixed-reward Cartpole sample-efficiency benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from config.off_policy import (
    DirectQCfg,
    PmdCfg,
    SacCfg,
    SuccessorQCfg,
)

AlgorithmName = Literal["ppo", "sac", "pmd"]


def cartpole_sac_cfg() -> SacCfg:
    """Stable local SAC control with a direct scalar-Q ensemble."""
    return SacCfg(
        value=DirectQCfg(num_heads=5),
        initial_entropy_temperature=0.006,
        automatic_entropy_tuning=True,
    )


def cartpole_pmd_cfg() -> PmdCfg:
    """Exponential PMD; switch value to DirectQCfg for the direct-Q ablation."""
    return PmdCfg(
        value=SuccessorQCfg(num_heads=5),
        beta=1.0,
        candidate_actions=64,
        candidate_target_kl=0.05,
        actor_projection_kl=0.02,
        entropy_temperature=0.0,
    )


@dataclass(slots=True)
class CartpoleBenchmarkCfg:
    """Shared experiment choices; simulator transitions are the sample unit."""

    algorithm: AlgorithmName
    seed: int = 1
    num_envs: int = 256
    transition_budget: int = 500_000
    evaluation_interval_transitions: int = 65_536
    evaluation_steps: int = 250
    device: str = "cuda:0"
    project: str = "isaac-irl-cartpole-sample-efficiency"
    group: str = "cartpole-fixed-reward-v4"
    run_name: str | None = None
    log_root: str = "logs/cartpole_benchmark"
    record_video: bool = False
    video_length: int = 250
    action_clip: float = 1.0
    sac: SacCfg = field(default_factory=cartpole_sac_cfg)
    pmd: PmdCfg = field(default_factory=cartpole_pmd_cfg)

    def __post_init__(self) -> None:
        if self.num_envs <= 0:
            raise ValueError("num_envs must be > 0.")
        if self.transition_budget <= 0:
            raise ValueError("transition_budget must be > 0.")
        if self.evaluation_interval_transitions <= 0:
            raise ValueError("evaluation_interval_transitions must be > 0.")
        if self.evaluation_steps <= 0:
            raise ValueError("evaluation_steps must be > 0.")
        if self.action_clip <= 0.0:
            raise ValueError("action_clip must be > 0.")
        # Keep vector rollout size aligned with the requested benchmark.
        self.sac = replace(
            self.sac,
            policy=replace(self.sac.policy, action_bound=self.action_clip),
        )
        self.pmd = replace(
            self.pmd,
            policy=replace(self.pmd.policy, action_bound=self.action_clip),
        )
