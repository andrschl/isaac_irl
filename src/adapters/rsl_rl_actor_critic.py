"""Construct rsl_rl 5.x actor + critic `MLPModel`s from an `ActorCriticCfg`.

Translates the legacy (init_noise_std, noise_std_type, state_dependent_std)
trio that the YAML still uses into the `distribution_cfg` dict consumed by
rsl_rl's `MLPModel`. Only Gaussian (non-state-dependent) is supported.
"""

from __future__ import annotations

from typing import Any

from config.train_irl import ActorCriticCfg


def make_actor_critic(
    *,
    obs: Any,
    obs_groups: dict[str, list[str]],
    action_dim: int,
    cfg: ActorCriticCfg,
    device: str,
    MLPModel: Any,
) -> tuple[Any, Any]:
    """Build (actor, critic) `MLPModel` instances on ``device``."""
    if cfg.state_dependent_std:
        raise NotImplementedError(
            "state_dependent_std=True is not wired into the new MLPModel construction yet. "
            "Add a HeteroscedasticGaussianDistribution branch if you need it."
        )
    distribution_cfg = {
        "class_name": "rsl_rl.modules.GaussianDistribution",
        "init_std": float(cfg.init_noise_std),
        "std_type": str(cfg.noise_std_type),
    }
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        int(action_dim),
        hidden_dims=list(cfg.actor_hidden_dims),
        activation=str(cfg.activation),
        obs_normalization=bool(cfg.actor_obs_normalization),
        distribution_cfg=distribution_cfg,
    ).to(device)
    critic = MLPModel(
        obs,
        obs_groups,
        "critic",
        1,
        hidden_dims=list(cfg.critic_hidden_dims),
        activation=str(cfg.activation),
        obs_normalization=bool(cfg.critic_obs_normalization),
        distribution_cfg=None,
    ).to(device)
    return actor, critic
