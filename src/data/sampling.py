"""Generic episode-list helpers: subsetting and train/val splits."""

from __future__ import annotations

from typing import TypeVar

import torch

_EpisodeT = TypeVar("_EpisodeT")


def subset_episodes(
    episodes: list[_EpisodeT],
    max_num: int,
    strategy: str,
    seed: int,
) -> list[_EpisodeT]:
    """Return up to ``max_num`` episodes. ``strategy``: 'first' | 'random'."""
    if len(episodes) <= max_num:
        return episodes
    if strategy == "first":
        return episodes[:max_num]
    if strategy == "random":
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(episodes), generator=rng)[:max_num].tolist()
        return [episodes[i] for i in indices]
    raise ValueError(
        f"Unknown expert_subset_strategy '{strategy}'. Expected 'first' or 'random'."
    )


def split_holdout_episodes(
    episodes: list[_EpisodeT],
    val_fraction: float,
    seed: int,
) -> tuple[list[_EpisodeT], list[_EpisodeT]]:
    """Split episodes into ``(train, val)`` by a held-out *episode* fraction.

    The split is at the episode level (not per-transition) so no trajectory
    leaks across train/val. ``val_fraction`` is rounded down but always leaves
    at least one episode in each side when the split is requested and possible.
    """
    n = len(episodes)
    if val_fraction <= 0.0 or n < 2:
        return episodes, []
    n_val = max(1, min(n - 1, int(n * val_fraction)))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    val_idx = set(perm[:n_val])
    val = [episodes[i] for i in range(n) if i in val_idx]
    train = [episodes[i] for i in range(n) if i not in val_idx]
    return train, val
