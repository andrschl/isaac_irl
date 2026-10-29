"""Reward features read from Isaac Lab's reward manager, as the runner sees them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from config.schema import FeatureMapCfg
from interfaces import FeatureSpec
from reward.manager_terms import SelectedTerm, select_reward_terms


class ManagerBasedFeatureSource:
    """``FeatureSource`` over the environment's own reward terms.

    The feature order is fixed at construction by reading the reward manager's
    term list - no term is evaluated to learn its name, so building this does
    not step or probe the environment. Per-step extraction then evaluates
    exactly those terms against the current env state.
    """

    def __init__(
        self,
        env: Any,
        cfg: FeatureMapCfg | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.cfg = cfg or FeatureMapCfg()
        self.device = device
        self.terms = select_reward_terms(
            env.unwrapped.reward_manager,
            ignored_terms=self.cfg.ignored_reward_terms,
            force_include_terms=self.cfg.force_include_terms,
        )
        self.spec = FeatureSpec(
            dim=len(self.terms), names=tuple(term.name for term in self.terms)
        )

    def features(self, env: Any) -> torch.Tensor:
        """Evaluate every feature term against the current state. Shape: [N, D]."""
        return manager_based_reward_features(env, terms=self.terms, device=self.device)


def manager_based_reward_features(
    env: Any,
    *,
    terms: tuple[SelectedTerm, ...],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Stack the selected reward terms into a feature matrix ``[N, D]``."""
    unwrapped = env.unwrapped
    num_envs = int(unwrapped.num_envs)
    columns = [term.evaluate(unwrapped, num_envs) for term in terms]
    features = torch.stack(columns, dim=1)
    return features if device is None else features.to(device)


@dataclass(frozen=True, slots=True)
class FeatureColumns:
    """A named subset of a collected feature vector, in a fixed order.

    Replay and the critics carry every tapped reward term. The reward learner
    may fit a named subset: mixed sparse+learned reward fixes terminal success
    and excludes it, while pure learned reward can include it explicitly. The
    same columns reach IRL, validation feature means, and their labels.

    Shapes: ``apply`` maps ``[..., D]`` to ``[..., len(names)]``.
    """

    source_names: tuple[str, ...]
    names: tuple[str, ...]
    columns: torch.Tensor
    """[len(names)] positions in the source vector, as int64."""

    @classmethod
    def select(
        cls, source_names: Sequence[str], names: Sequence[str]
    ) -> "FeatureColumns":
        """Resolve ``names`` against ``source_names``, rejecting any it lacks."""
        source = tuple(str(name) for name in source_names)
        wanted = tuple(str(name) for name in names)
        if len(set(source)) != len(source):
            raise ValueError(f"Collected feature names must be unique, got {source}.")
        missing = [name for name in wanted if name not in source]
        if missing:
            raise ValueError(
                f"Reward features {missing} are not collected; the environment "
                f"reports {list(source)}."
            )
        return cls(
            source_names=source,
            names=wanted,
            columns=torch.tensor([source.index(name) for name in wanted]),
        )

    @classmethod
    def identity(cls, source_names: Sequence[str]) -> "FeatureColumns":
        """Keep every collected column, in the order it was collected."""
        return cls.select(source_names, source_names)

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def is_identity(self) -> bool:
        return self.names == self.source_names

    def apply(self, features: torch.Tensor) -> torch.Tensor:
        """Select the reward-learning columns of ``features`` [..., D]."""
        if features.shape[-1] != len(self.source_names):
            raise ValueError(
                f"Expected features [...,{len(self.source_names)}] to match "
                f"{list(self.source_names)}, got {tuple(features.shape)}."
            )
        if self.is_identity:
            return features
        return features.index_select(-1, self.columns.to(features.device))

    def spec(self) -> FeatureSpec:
        return FeatureSpec(dim=self.dim, names=self.names)
