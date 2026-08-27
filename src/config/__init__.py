"""Experiment configuration dataclasses and YAML parsing."""

from __future__ import annotations

from typing import Any

from .fixed_reward import FixedRewardRunnerCfg
from .off_policy import (
    DirectQCfg,
    OffPolicyAlgorithmCfg,
    OffPolicyTrainCfg,
    PmdCfg,
    SacCfg,
    SuccessorQCfg,
    TanhGaussianPolicyCfg,
    TransitionReplayCfg,
)

_TRAIN_EXPORTS = {
    "ActorCriticCfg",
    "EnvCfg",
    "PpoAlgoCfg",
    "TrainCfg",
    "load_train_cfg",
}


def __getattr__(name: str) -> Any:
    """Load the main training schema lazily to avoid config/algorithm cycles."""
    if name not in _TRAIN_EXPORTS:
        raise AttributeError(name)
    from . import train_irl

    return getattr(train_irl, name)


__all__ = [
    "ActorCriticCfg",
    "DirectQCfg",
    "EnvCfg",
    "FixedRewardRunnerCfg",
    "OffPolicyAlgorithmCfg",
    "OffPolicyTrainCfg",
    "PmdCfg",
    "PpoAlgoCfg",
    "SacCfg",
    "SuccessorQCfg",
    "TanhGaussianPolicyCfg",
    "TrainCfg",
    "TransitionReplayCfg",
    "load_train_cfg",
]
