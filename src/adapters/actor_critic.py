"""Construct rsl_rl 5.x actor + critic `MLPModel`s from an `ActorCriticCfg`.

Translates the legacy (init_noise_std, noise_std_type)
trio that the YAML still uses into the `distribution_cfg` dict consumed by
rsl_rl's `MLPModel`. Only Gaussian (non-state-dependent) is supported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.schema import ActorCriticCfg


def make_actor(
    *,
    obs: Any,
    obs_groups: dict[str, list[str]],
    action_dim: int,
    cfg: "ActorCriticCfg",
    device: str,
    MLPModel: Any,
) -> Any:
    """Build only the Gaussian actor; no scalar critic is constructed."""
    distribution_cfg = {
        "class_name": "rsl_rl.modules.GaussianDistribution",
        "init_std": float(cfg.init_noise_std),
        "std_type": str(cfg.noise_std_type),
    }
    return MLPModel(
        obs,
        obs_groups,
        "actor",
        int(action_dim),
        hidden_dims=list(cfg.actor_hidden_dims),
        activation=str(cfg.activation),
        obs_normalization=bool(cfg.actor_obs_normalization),
        distribution_cfg=distribution_cfg,
    ).to(device)


def make_actor_critic(
    *,
    obs: Any,
    obs_groups: dict[str, list[str]],
    action_dim: int,
    cfg: "ActorCriticCfg",
    device: str,
    MLPModel: Any,
) -> tuple[Any, Any]:
    """Build (actor, critic) `MLPModel` instances on ``device``."""
    actor = make_actor(
        obs=obs,
        obs_groups=obs_groups,
        action_dim=action_dim,
        cfg=cfg,
        device=device,
        MLPModel=MLPModel,
    )
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
