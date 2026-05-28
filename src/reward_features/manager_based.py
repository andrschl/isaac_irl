from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch

from interfaces import FeatureSpec


@dataclass
class ManagerBasedFeatureCfg:
    """Configuration for extracting features from Isaac Lab's reward manager terms."""

    ignored_reward_terms: set[str] = field(default_factory=set)
    force_include_terms: set[str] = field(default_factory=set)
    """Names of reward-manager terms to extract as features even when their
    env-reward weight is 0. Use for terms injected purely as IRL features
    (e.g. ``success_bonus`` with weight=0) that should not also contribute to
    the env reward signal."""


def _as_term_vector(term_value: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Normalize a reward term output into shape [num_envs]."""
    if not isinstance(term_value, torch.Tensor):
        term_value = torch.as_tensor(term_value)

    if term_value.ndim == 0:
        return term_value.repeat(num_envs)
    if term_value.ndim == 1:
        if term_value.shape[0] != num_envs:
            raise ValueError(
                f"Reward term returned shape {tuple(term_value.shape)}; expected first dim {num_envs}."
            )
        return term_value

    # Many manager terms return [N, 1]; flatten that case to [N].
    if term_value.shape[0] == num_envs and term_value.numel() == num_envs:
        return term_value.reshape(num_envs)

    raise ValueError(
        f"Unsupported reward term output shape {tuple(term_value.shape)}. "
        "Expected scalar, [N], or [N, 1]."
    )


def manager_based_reward_feature_dict(
    env,
    ignored_reward_terms: Iterable[str] = (),
    device: torch.device | str | None = None,
    force_include_terms: Iterable[str] = (),
) -> dict[str, torch.Tensor]:
    """
    Extract reward-manager term values as a named feature mapping.

    By default skips reward terms with weight=0 (disabled). Terms listed in
    ``force_include_terms`` are extracted regardless of weight — used for
    feature-only terms injected with weight=0 (e.g. ``success_bonus``).

    Returns:
        feature_values:
            Mapping of `term_name -> feature_value`.
            Each feature value has shape [N], where N is number of envs.
    """
    unwrapped_env = env.unwrapped
    reward_manager = unwrapped_env.reward_manager
    ignored = set(ignored_reward_terms)
    forced = set(force_include_terms)
    num_envs = int(unwrapped_env.num_envs)

    feature_values: dict[str, torch.Tensor] = {}
    for name, term_cfg in zip(reward_manager._term_names, reward_manager._term_cfgs):
        if name in ignored:
            continue
        if term_cfg.weight == 0.0 and name not in forced:
            continue

        term_params = term_cfg.params or {}
        term_value = term_cfg.func(unwrapped_env, **term_params)
        term_vector = _as_term_vector(term_value, num_envs)
        if device is not None:
            term_vector = term_vector.to(device)
        feature_values[str(name)] = term_vector

    return feature_values


def manager_based_reward_features(
    env,
    ignored_reward_terms: Iterable[str] = (),
    device: torch.device | str | None = None,
    force_include_terms: Iterable[str] = (),
) -> torch.Tensor:
    """Extract reward-manager term values as a feature matrix [num_envs, num_terms]."""
    named_features = manager_based_reward_feature_dict(
        env=env,
        ignored_reward_terms=ignored_reward_terms,
        device=device,
        force_include_terms=force_include_terms,
    )

    num_envs = int(env.unwrapped.num_envs)
    if not named_features:
        return torch.empty((num_envs, 0), dtype=torch.float32, device=device)

    ordered_values = [named_features[name] for name in named_features.keys()]
    return torch.stack(ordered_values, dim=1)


class ManagerBasedFeatureSource:
    """``FeatureSource`` over Isaac Lab's reward manager.

    Probes the env's reward manager once at construction to fix the feature
    order (and learn the feature names) and exposes that via ``spec``.
    Isaac's reward manager already owns the env state, so per-step extraction
    only needs the env handle.
    """

    def __init__(
        self,
        env,
        cfg: ManagerBasedFeatureCfg | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.cfg = cfg or ManagerBasedFeatureCfg()
        self.device = device
        feature_dict = manager_based_reward_feature_dict(
            env=env,
            ignored_reward_terms=self.cfg.ignored_reward_terms,
            device=device,
            force_include_terms=self.cfg.force_include_terms,
        )
        names = tuple(feature_dict.keys())
        self.spec = FeatureSpec(dim=len(names), names=names)

    def features(self, env) -> torch.Tensor:
        """Extract `[num_envs, D]` features from the current env state."""
        return manager_based_reward_features(
            env=env,
            ignored_reward_terms=self.cfg.ignored_reward_terms,
            device=self.device,
            force_include_terms=self.cfg.force_include_terms,
        )
