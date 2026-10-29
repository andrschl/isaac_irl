"""The three learners: a reward (IRL), a policy (RL), and a regularizer (BC)."""

from .bc import BC, build_bc_alg
from .irl import IrlAlgorithm
from .off_policy.pmd import Pmd
from .on_policy.ppo_bc import make_ppo_with_bc_cls

__all__ = [
    "BC",
    "IrlAlgorithm",
    "Pmd",
    "build_bc_alg",
    "make_ppo_with_bc_cls",
]
