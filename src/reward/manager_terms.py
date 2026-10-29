"""Reading reward terms off Isaac Lab's reward manager.

The selection rule lives here once. Two things read reward terms - the online
feature source and the pre-reset tap that records demonstrations - and if they
disagreed about which terms count, recorded demonstrations and online features
would describe different reward spaces while looking identical.

This is also the only module that touches the reward manager's private term
lists, so the coupling to Isaac's internals is one import away from everything
else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True, slots=True)
class SelectedTerm:
    """One reward term that counts as a feature.

    ``index`` is the term's position in the reward manager's own order, which
    is also its column in the per-step reward matrix. ``weight`` is the weight
    at selection time; a term selected at weight zero is a feature only and has
    no column in that matrix.
    """

    name: str
    weight: float
    index: int
    cfg: Any

    @property
    def contributes_to_reward(self) -> bool:
        return self.weight != 0.0

    def evaluate(self, env: Any, num_envs: int) -> torch.Tensor:
        """Compute this term directly from env state. Shape: [N]."""
        return as_term_vector(self.cfg.func(env, **(self.cfg.params or {})), num_envs)


def select_reward_terms(
    reward_manager: Any,
    *,
    ignored_terms: Iterable[str] = (),
    force_include_terms: Iterable[str] = (),
) -> tuple[SelectedTerm, ...]:
    """Choose the feature terms, in reward-manager order.

    Skip terms named in ``ignored_terms``, and skip zero-weight terms unless
    they were explicitly requested - a term injected purely as a feature (a
    success bonus at weight zero) has to be asked for, or it would be
    indistinguishable from a term that is switched off.
    """
    ignored = set(ignored_terms)
    forced = set(force_include_terms)
    terms = [
        SelectedTerm(name=str(name), weight=float(cfg.weight), index=index, cfg=cfg)
        for index, (name, cfg) in enumerate(
            zip(reward_manager._term_names, reward_manager._term_cfgs, strict=True)
        )
        if str(name) not in ignored
        and (float(cfg.weight) != 0.0 or str(name) in forced)
    ]
    if not terms:
        raise ValueError(
            "The reward manager exposes no feature terms: every term is either "
            f"ignored {sorted(ignored)} or has weight zero without being "
            f"requested as a feature {sorted(forced)}."
        )
    return tuple(terms)


def as_term_vector(term_value: Any, num_envs: int) -> torch.Tensor:
    """Normalize one reward term's output to ``[N]``."""
    if not isinstance(term_value, torch.Tensor):
        term_value = torch.as_tensor(term_value)
    if term_value.ndim == 0:
        return term_value.repeat(num_envs)
    if term_value.ndim == 1 and term_value.shape[0] == num_envs:
        return term_value
    # Many manager terms return [N, 1].
    if tuple(term_value.shape) == (num_envs, 1):
        return term_value.reshape(num_envs)
    raise ValueError(
        f"Reward term returned shape {tuple(term_value.shape)}; expected a "
        f"scalar, [{num_envs}], or [{num_envs}, 1]."
    )
