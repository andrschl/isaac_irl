"""Algorithm entry points for local training scripts."""

from .bc import BC, BCCfg
from .irl import IRLCfg, IrlAlgorithm
from .ppo_with_bc import make_ppo_with_bc_cls

__all__ = ["BC", "BCCfg", "IRLCfg", "IrlAlgorithm", "make_ppo_with_bc_cls"]
