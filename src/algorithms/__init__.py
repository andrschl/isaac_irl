"""Algorithm entry points for local training scripts."""

from .bc import BC, BCCfg, build_bc_alg
from .irl import IRLCfg, IrlAlgorithm
from .ppo_with_bc import make_ppo_with_bc_cls

__all__ = ["BC", "BCCfg", "IRLCfg", "IrlAlgorithm", "build_bc_alg", "make_ppo_with_bc_cls"]
