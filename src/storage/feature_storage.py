from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.utils.rnn as rnn_utils
from einops import einsum, rearrange

from reward_model.base import BaseRewardModel
from utils.runtime_context import RuntimeContext


@dataclass(slots=True)
class FeatureBufCfg:
    """Buffer-specific configuration (chosen by the experiment)."""

    capacity_steps: int = 1_000_000
    store_device: torch.device | str = "cpu"
    store_dtype: torch.dtype = torch.float16
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

        self._store_device = torch.device(cfg.store_device)

        # Current unfinished episodes, one list per env. Each entry is a [D] tensor.
        self._cur: list[list[torch.Tensor]] = [[] for _ in range(ctx.num_envs)]

        # Completed episodes and their lengths.
        self._eps: deque[torch.Tensor] = deque()   # each episode: [T, D]
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

        # Move once to storage device/dtype.
        z_store = z.detach().to(device=self._store_device, dtype=self.cfg.store_dtype)
        done_cpu = done.detach().to("cpu").bool()

        for i in range(self.ctx.num_envs):
            # clone() avoids retaining a view into the full [N,D] tensor
            self._cur[i].append(z_store[i].clone())
            if bool(done_cpu[i].item()):
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
            done = torch.tensor([t == t_steps - 1], dtype=torch.bool, device=feats.device)
            self.add_step(feats[t : t + 1], done)

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
            return torch.multinomial(probs, num_samples=batch_size, replacement=True).tolist()

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
        feats, mask, lengths = self.sample_episodes(batch_size=batch_size, device=device)
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
    def discounted_returns(r: torch.Tensor, mask: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        r: [B,T], mask: [B,T] -> returns [B]
        """
        _, t = r.shape
        powers = gamma ** torch.arange(t, device=r.device, dtype=r.dtype)
        return einsum(r, mask.to(dtype=r.dtype), powers, "b t, b t, t -> b")

    @staticmethod
    def eval_discounted_returns_from_model(
        feats: torch.Tensor,
        mask: torch.Tensor,
        reward_model: BaseRewardModel,
        gamma: float,
    ) -> torch.Tensor:
        """
        Evaluate discounted returns from a reward model.

        Args:
            feats: [B, T, D]
            mask: [B, T]
            reward_model: feature-based reward model
            gamma: discount factor

        Returns:
            returns: [B]
        """
        reward_param = next(reward_model.parameters(), None)
        if reward_param is None:
            model_device = feats.device
            model_dtype = feats.dtype
        else:
            model_device = reward_param.device
            model_dtype = reward_param.dtype

        if bool(reward_model.is_linear):
            discounted_feats = FeatureTrajectoryBuffer.discounted_feature_returns(feats, mask, gamma)
            discounted_feats = discounted_feats.to(device=model_device, dtype=model_dtype)
            rewards = reward_model.get_reward_from_features(discounted_feats)
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.as_tensor(rewards, device=model_device, dtype=model_dtype)
            if rewards.ndim == 2 and rewards.shape[-1] == 1:
                rewards = rearrange(rewards, "b 1 -> b")
            return rewards

        feats = feats.to(device=model_device, dtype=model_dtype)
        mask = mask.to(device=model_device, dtype=torch.bool)
        rewards = reward_model.get_reward_from_features(feats, mask)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards, device=model_device, dtype=model_dtype)
        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rewards = rearrange(rewards, "b t 1 -> b t")
        return FeatureTrajectoryBuffer.discounted_returns(rewards, mask, gamma)

    @torch.no_grad()
    def sample_and_eval_returns(
        self,
        reward_model: BaseRewardModel,
        batch_size: int,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        """
        Convenience helper for debugging/evaluation.

        Returns:
            feats   [B, T, D]
            mask    [B, T]
            lengths [B]
            returns [B]
        """
        feats, mask, lengths = self.sample_episodes(batch_size=batch_size, device=device)
        returns = self.eval_discounted_returns_from_model(
            feats=feats,
            mask=mask,
            reward_model=reward_model,
            gamma=self.gamma,
        )
        return {
            "feats": feats,
            "mask": mask,
            "lengths": lengths,
            "returns": returns,
        }
