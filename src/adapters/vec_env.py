"""Adapter that exposes ``action_dim`` on top of an rsl_rl-shaped VecEnv.

The core's :class:`interfaces.VecEnv` protocol calls this field ``action_dim``;
rsl_rl's :class:`rsl_rl.env.vec_env.VecEnv` calls it ``num_actions``. The
translation lives here so the rest of `src/` never spells the legacy name.
Other attributes (``num_envs``, ``device``, ``max_episode_length``,
``get_observations``, ``step``, ``episode_length_buf`` …) proxy through
unchanged, in both directions.
"""

from __future__ import annotations

from typing import Any


class RslRlVecEnvAdapter:
    """Thin pass-through wrapper that surfaces ``action_dim``."""

    def __init__(self, env: Any) -> None:
        object.__setattr__(self, "_env", env)

    @property
    def action_dim(self) -> int:
        return int(self._env.num_actions)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal attribute lookup misses, so
        # `action_dim` (a property) takes precedence. Everything else,
        # including `num_actions`, proxies straight to the wrapped env.
        return getattr(self._env, name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Writes have to proxy too. Without this, assigning through the
        # adapter creates an attribute *on the adapter* that shadows the
        # environment's own, so the read still succeeds and the environment
        # never sees the value - which is how staggered episode ages were
        # silently dropped. `_env` is the adapter's, everything else is not.
        if name == "_env":
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)
