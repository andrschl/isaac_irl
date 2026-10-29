"""Episode-structured reward features, the data a reward is learned from."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.utils.rnn as rnn_utils
from einops import einsum, rearrange

from config.schema import FeatureBufCfg


class FeatureTrajectoryBuffer:
    """Completed feature episodes, stored as ragged ``[T, D]`` tensors.

    Episodes arrive already batched - a policy generation as ``[N, T, D]``, a
    recorded demonstration as ``[T, D]`` - so ingestion is vectorized and never
    walks environments or timesteps in Python.

    Sampling pads to the longest episode in the batch and returns the mask that
    says which steps are real, because a discounted feature return has to stop
    where its episode did.
    """

    def __init__(
        self,
        cfg: FeatureBufCfg,
        *,
        feature_dim: int,
        gamma: float,
        device: str | torch.device = "cpu",
    ) -> None:
        if feature_dim <= 0:
            raise ValueError(f"`feature_dim` must be > 0, got {feature_dim}.")
        self.cfg = cfg
        self.feature_dim = int(feature_dim)
        self.gamma = float(gamma)
        self._store_device = torch.device(cfg.resolve_store_device(device))
        self._episodes: deque[torch.Tensor] = deque()
        self._lengths: deque[int] = deque()
        self._steps_stored = 0

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def steps_stored(self) -> int:
        return self._steps_stored

    def clear(self) -> None:
        self._episodes.clear()
        self._lengths.clear()
        self._steps_stored = 0

    def append_episode(self, features: torch.Tensor) -> None:
        """Store one complete episode. Shapes: features [T, D]."""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features [T, {self.feature_dim}], got "
                f"{tuple(features.shape)}."
            )
        episode_length = int(features.shape[0])
        if episode_length < self.cfg.min_ep_len:
            return
        self._episodes.append(
            features.detach()
            .to(device=self._store_device, dtype=self.cfg.store_dtype)
            .clone()
        )
        self._lengths.append(episode_length)
        self._steps_stored += episode_length
        self._evict_by_capacity()

    def add_rollout(self, features: torch.Tensor, dones: torch.Tensor) -> None:
        """Store the episodes a rollout contains. Shapes: [N,T,D], [N,T].

        A fixed-horizon rollout cuts the last episode of every environment
        short; that trailing segment is kept, because a rollout shorter than an
        episode would otherwise store nothing at all. Discounting never crosses
        a reset, which is what makes an imitator return comparable to an expert
        demonstration's.
        """
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected features [N,T,{self.feature_dim}], got "
                f"{tuple(features.shape)}."
            )
        if tuple(dones.shape) != tuple(features.shape[:2]) or dones.dtype != torch.bool:
            raise ValueError(
                f"Expected bool dones {tuple(features.shape[:2])}, got "
                f"{tuple(dones.shape)} with dtype {dones.dtype}."
            )
        horizon = int(features.shape[1])
        for row, row_dones in zip(features, dones, strict=True):
            boundaries = (row_dones.nonzero(as_tuple=False).flatten() + 1).tolist()
            if not boundaries or boundaries[-1] != horizon:
                boundaries.append(horizon)
            start = 0
            for end in boundaries:
                self.append_episode(row[start:end])
                start = end

    def replace_episode_batch(self, feats: torch.Tensor, mask: torch.Tensor) -> None:
        """Replace everything stored with one batch of episodes.

        Shapes: feats [B, T, D], mask [B, T] as a contiguous valid prefix.
        """
        if feats.ndim != 3 or feats.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected feats [B, T, {self.feature_dim}], got {tuple(feats.shape)}."
            )
        if tuple(mask.shape) != tuple(feats.shape[:2]) or mask.dtype != torch.bool:
            raise ValueError(
                f"Expected a bool mask {tuple(feats.shape[:2])}, got "
                f"{tuple(mask.shape)} with dtype {mask.dtype}."
            )
        prefix = mask.to(torch.int64).cummin(dim=1).values.to(torch.bool)
        if not torch.equal(mask, prefix):
            raise ValueError(
                "mask must be a contiguous valid prefix for every episode."
            )
        self.clear()
        stored = feats.detach().to(
            device=self._store_device, dtype=self.cfg.store_dtype
        )
        for episode, length in zip(stored, mask.sum(dim=1).tolist(), strict=True):
            self.append_episode(episode[: int(length)])

    def _evict_by_capacity(self) -> None:
        while self._steps_stored > self.cfg.capacity_steps and self._episodes:
            self._episodes.popleft()
            self._steps_stored -= int(self._lengths.popleft())

    def sample_episodes(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample episodes, padded to the longest one drawn.

        Returns feats [B, T_max, D], mask [B, T_max], and lengths [B].
        """
        indices = self._sample_indices(batch_size)
        device = torch.device(device)
        episodes = [
            self._episodes[index].to(device=device, dtype=self.cfg.store_dtype)
            for index in indices
        ]
        lengths = torch.tensor(
            [episode.shape[0] for episode in episodes], device=device
        )
        feats = rnn_utils.pad_sequence(episodes, batch_first=True)
        time_index = torch.arange(feats.shape[1], device=device)
        mask = rearrange(time_index, "t -> 1 t") < rearrange(lengths, "b -> b 1")
        return feats, mask, lengths

    def mean_feature_return(
        self,
        *,
        gamma: float,
        device: torch.device | str,
        normalize_by_episode_length: bool = False,
    ) -> torch.Tensor:
        """Mean over every stored episode of its feature return [D]."""
        if not self._episodes:
            raise RuntimeError("Cannot average: no completed episodes are stored.")
        target_device = torch.device(device)
        returns = []
        for episode in self._episodes:
            features = episode.to(device=target_device, dtype=torch.float32)
            powers = gamma ** torch.arange(
                features.shape[0], device=target_device, dtype=features.dtype
            )
            feature_return = (features * rearrange(powers, "t -> t 1")).sum(0)
            if normalize_by_episode_length:
                feature_return = feature_return / features.shape[0]
            returns.append(feature_return)
        return torch.stack(returns).mean(0)

    def _sample_indices(self, batch_size: int) -> list[int]:
        if not self._episodes:
            raise RuntimeError("Cannot sample: no completed episodes are stored.")
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {batch_size}.")
        if not self.cfg.sample_weighted_by_length:
            return torch.randint(0, len(self._episodes), (batch_size,)).tolist()
        # Weighting by length samples timesteps uniformly rather than episodes,
        # so a long episode is not compressed into the same weight as a short one.
        lengths = torch.tensor(list(self._lengths), dtype=torch.float32)
        return torch.multinomial(
            lengths / lengths.sum(), num_samples=batch_size, replacement=True
        ).tolist()

    @staticmethod
    def discounted_feature_returns(
        feats: torch.Tensor, mask: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """Discounted per-feature return. Shapes: [B,T,D], [B,T] -> [B,D]."""
        horizon = feats.shape[1]
        powers = gamma ** torch.arange(horizon, device=feats.device, dtype=feats.dtype)
        return einsum(feats, mask.to(dtype=feats.dtype), powers, "b t d, b t, t -> b d")
