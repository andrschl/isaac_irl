"""Algorithm entry points for the retained PPO, SAC, and PMD stack."""

from .bc import BC, BCCfg, build_bc_alg
from .exponential_policy import (
    ExponentialTarget,
    adaptive_exponential_target,
    exponential_log_weights,
    exponential_target,
)
from .irl import IrlAlgorithm, IRLCfg
from .pmd import Pmd
from .ppo_with_bc import make_ppo_with_bc_cls
from .sac import Sac
from .scalar_action_value import ScalarActionValue, ScalarActionValueEnsemble
from .soft_value import entropy_return_td_target, soft_task_q_td_target
from .successor_ensemble import SuccessorFeatureEnsemble

__all__ = [
    "BC",
    "BCCfg",
    "ExponentialTarget",
    "IRLCfg",
    "IrlAlgorithm",
    "Pmd",
    "Sac",
    "ScalarActionValue",
    "ScalarActionValueEnsemble",
    "SuccessorFeatureEnsemble",
    "adaptive_exponential_target",
    "build_bc_alg",
    "entropy_return_td_target",
    "exponential_log_weights",
    "exponential_target",
    "make_ppo_with_bc_cls",
    "soft_task_q_td_target",
]
