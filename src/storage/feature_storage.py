from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.utils.rnn as rnn_utils
from einops import einsum, rearrange

from utils.runtime_context import RuntimeContext


@dataclass(slots=True)
class FeatureBufCfg:
    """Buffer-specific configuration (chosen by the experiment)."""

    capacity_steps: int = 1_000_000
    # `None` -> resolved to `ctx.device` in FeatureTrajectoryBuffer.__init__.
    # Imitator buffers default to ctx.device so the per-step append stays on
    # GPU; long-lived expert buffers should set "cpu" explicitly to keep
    # episode storage off the device.
    store_device: torch.device | str | None = None
    store_dtype: torch.dtype = torch.float32
    min_ep_len: int = 2
    sample_weighted_by_length: bool = False


class FeatureTrajectoryBuffer:
    """
    Vectorized, features-only episode buffer.

    Stores completed episodes as tensors [T, D] in a deque.
    Supports:
      - add_step(z[N,D], done[N])
      - sample_episodes(...) -> feats [B,T,D], mask [B,T], lengths [B]
      - sample_discounted_feature_returns(...) -> [B,D], lengths [B]
      - discounted_returns(...) for reward sequences
    """

    def __init__(self, cfg: FeatureBufCfg, ctx: RuntimeContext, gamma: float) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self.gamma = float(gamma)

        store_device = cfg.store_device if cfg.store_device is not None else ctx.device
        self._store_device = torch.device(store_device)

        # Current unfinished episodes, one list per env. Each entry is a [D] tensor.
        self._cur: list[list[torch.Tensor]] = [[] for _ in range(ctx.num_envs)]

        # Completed episodes and their lengths.
        self._eps: deque[torch.Tensor] = deque()  # each episode: [T, D]
        self._lengths: deque[int] = deque()

        self._steps_stored = 0

    def __len__(self) -> int:
        return len(self._eps)

    @property
    def steps_stored(self) -> int:
        return self._steps_stored

    def clear(self) -> None:
        self._cur = [[] for _ in range(self.ctx.num_envs)]
        self._eps.clear()
        self._lengths.clear()
        self._steps_stored = 0

    def add_step(self, z: torch.Tensor, done: torch.Tensor) -> None:
        """
        Add one vectorized env step.

        Args:
            z:    [N, D] feature vectors for all envs at current timestep
            done: [N] bool, True if the episode ended after this step
        """
        if z.ndim != 2:
            raise ValueError(f"`z` must be [N,D], got {tuple(z.shape)}")
        if done.ndim != 1:
            raise ValueError(f"`done` must be [N], got {tuple(done.shape)}")

        n, d = z.shape
        if n != self.ctx.num_envs or d != self.ctx.feature_dim:
            raise ValueError(
                f"Expected z shape [{self.ctx.num_envs}, {self.ctx.feature_dim}], got {tuple(z.shape)}"
            )
        if done.shape[0] != self.ctx.num_envs:
            raise ValueError(
                f"Expected done shape [{self.ctx.num_envs}], got {tuple(done.shape)}"
            )

        # Move once to storage device/dtype, then pull dones once to a Python
        # list so the per-env loop never triggers a GPU->CPU sync.
        z_store = z.detach().to(device=self._store_device, dtype=self.cfg.store_dtype)
        done_list = done.detach().to("cpu").bool().tolist()

        for i in range(self.ctx.num_envs):
            # clone() avoids retaining a view into the full [N,D] tensor
            self._cur[i].append(z_store[i].clone())
            if done_list[i]:
                self._finalize(i)

    def finalize_in_progress_episodes(self) -> None:
        """Force-flush each env's in-progress episode into the completed deque.

        Validation rolls out for a fixed horizon and typically does not see
        natural episode ends, so the buffer would otherwise hold zero
        completed episodes after the rollout. Callers that want fixed-length
        trajectories (one per env) should call this after the rollout loop.
        Respects ``cfg.min_ep_len``.
        """
        for i in range(self.ctx.num_envs):
            if self._cur[i]:
                self._finalize(i)

    def add_episode(self, feats: torch.Tensor) -> None:
        """
        Convenience helper for expert loading in a 1-env buffer.
        `feats` must be [T, D].
        """
        if self.ctx.num_envs != 1:
            raise ValueError("`add_episode` is only valid when ctx.num_envs == 1.")
        if feats.ndim != 2:
            raise ValueError(f"`feats` must be [T,D], got {tuple(feats.shape)}")
        if feats.shape[1] != self.ctx.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.ctx.feature_dim}, got {feats.shape[1]}"
            )

        t_steps = int(feats.shape[0])
        for t in range(t_steps):
            done = torch.tensor(
                [t == t_steps - 1], dtype=torch.bool, device=feats.device
            )
            self.add_step(feats[t : t + 1], done)

    def replace_episode_batch(self, feats: torch.Tensor, mask: torch.Tensor) -> None:
        """Replace storage from a dense trajectory batch.

        Args:
            feats: [B,T,D]
            mask: [B,T] bool contiguous-prefix validity mask
        """
        if feats.ndim != 3 or feats.shape[-1] != self.ctx.feature_dim:
            raise ValueError(
                f"Expected feats [B,T,{self.ctx.feature_dim}], got {tuple(feats.shape)}."
            )
        if tuple(mask.shape) != tuple(feats.shape[:2]) or mask.dtype != torch.bool:
            raise ValueError(
                f"Expected bool mask shape {tuple(feats.shape[:2])}, got "
                f"{tuple(mask.shape)} with dtype {mask.dtype}."
            )
        prefix = mask.to(torch.int64).cummin(dim=1).values.to(torch.bool)
        if not torch.equal(mask, prefix):
            raise ValueError(
                "mask must be a contiguous valid prefix for every episode."
            )
        self.clear()
        lengths = mask.sum(dim=1).to("cpu").tolist()
        for batch_idx, length in enumerate(lengths):
            episode_length = int(length)
            if episode_length < self.cfg.min_ep_len:
                continue
            episode = (
                feats[batch_idx, :episode_length]
                .detach()
                .to(device=self._store_device, dtype=self.cfg.store_dtype)
                .clone()
            )
            self._eps.append(episode)
            self._lengths.append(episode_length)
            self._steps_stored += episode_length
        self._evict_by_capacity()

    def _finalize(self, env_i: int) -> None:
        ep_list = self._cur[env_i]
        self._cur[env_i] = []

        ep_len = len(ep_list)
        if ep_len < self.cfg.min_ep_len:
            return

        ep = torch.stack(ep_list, dim=0)  # [T, D]
        self._eps.append(ep)
        self._lengths.append(ep_len)
        self._steps_stored += ep_len

        self._evict_by_capacity()

    def _evict_by_capacity(self) -> None:
        while self._steps_stored > self.cfg.capacity_steps and self._eps:
            _ = self._eps.popleft()
            old_len = self._lengths.popleft()
            self._steps_stored -= int(old_len)

    def _sample_indices(self, batch_size: int) -> list[int]:
        if len(self._eps) == 0:
            raise RuntimeError("No completed episodes in buffer.")
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}")

        if self.cfg.sample_weighted_by_length:
            lens = torch.tensor(list(self._lengths), dtype=torch.float32)
            probs = lens / lens.sum()
            return torch.multinomial(
                probs, num_samples=batch_size, replacement=True
            ).tolist()

        return torch.randint(0, len(self._eps), (batch_size,)).tolist()

    def sample_episodes(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            feats   [B, T_max, D]
            mask    [B, T_max] bool
            lengths [B]
        """
        idx = self._sample_indices(batch_size)
        device = torch.device(device)

        # Collect episodes on target device
        eps = [self._eps[i].to(device=device, dtype=self.cfg.store_dtype) for i in idx]
        lengths = torch.tensor([ep.shape[0] for ep in eps], device=device)

        feats = rnn_utils.pad_sequence(eps, batch_first=True)
        t_indices = torch.arange(feats.shape[1], device=device)
        mask = rearrange(t_indices, "t -> 1 t") < rearrange(lengths, "b -> b 1")

        return feats, mask, lengths

    def sample_discounted_feature_returns(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience for linear reward heads.

        Returns:
            discounted_feature_returns [B, D]
            lengths                    [B]
        """
        feats, mask, lengths = self.sample_episodes(
            batch_size=batch_size, device=device
        )
        z_ret = self.discounted_feature_returns(feats, mask, self.gamma)
        return z_ret, lengths

    @staticmethod
    def discounted_feature_returns(
        feats: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """
        feats: [B,T,D], mask: [B,T] -> discounted feature returns [B,D]
        """
        _, t, _ = feats.shape
        powers = gamma ** torch.arange(t, device=feats.device, dtype=feats.dtype)
        return einsum(feats, mask.to(dtype=feats.dtype), powers, "b t d, b t, t -> b d")

    @staticmethod
    def discounted_returns(
        r: torch.Tensor, mask: torch.Tensor, gamma: float
    ) -> torch.Tensor:
        """
        r: [B,T], mask: [B,T] -> returns [B]
        """
        _, t = r.shape
        powers = gamma ** torch.arange(t, device=r.device, dtype=r.dtype)
        return einsum(r, mask.to(dtype=r.dtype), powers, "b t, b t, t -> b")
