"""Backend adapters bridging rsl_rl-shaped objects to the core's protocols."""

from .rsl_rl_actor_critic import make_actor_critic
from .rsl_rl_ppo import RslRlPpoAdapter
from .rsl_rl_vec_env import RslRlVecEnvAdapter

__all__ = ["RslRlPpoAdapter", "RslRlVecEnvAdapter", "make_actor_critic"]
