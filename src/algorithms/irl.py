from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

from storage.feature_storage import FeatureBufCfg, FeatureTrajectoryBuffer
from reward_model.base import BaseRewardModel

# Adjust this import to your project path:
# e.g. from isaac_irl.runtime import RuntimeContext
from utils.runtime_context import RuntimeContext


@dataclass(slots=True)
class IRLCfg:
    """IRL-specific hyperparameters/config."""

    expert_data_path: str = ""
    batch_size: int = 256
    num_learning_epochs: int = 1
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    reward_loss_coef: float = 1.0
    reward_learning_rate: float | None = None
    discount_gamma: float | None = None
    normalize_returns_by_episode_length: bool = True


@dataclass(slots=True)
class Transition:
    """Lightweight cache for the current timestep."""

    actions: torch.Tensor | None = None
    features: torch.Tensor | None = None


class IRL:
    """
    IRL helper adapted to a feature-buffer pipeline.

    Key assumptions:
      - reward model operates on feature trajectories (feats, mask)
      - features come either from:
          * an env-based feature map: feature_map(env) -> [N, D]
          * or are passed explicitly into process_env_step(..., features=...)
      - expert/imitator trajectories are stored in FeatureTrajectoryBuffer
    """

    def __init__(
        self,
        rl_alg: Any,
        reward: BaseRewardModel,
        *,
        gamma: float,  # shared hyperparameter (single source of truth)
        cfg: IRLCfg | None = None,
        env: Any | None = None,
        feature_map: Callable[[Any], torch.Tensor] | None = None,  # env -> [N, D]
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.cfg = cfg or IRLCfg()

        # RL components
        self.rl_alg = rl_alg
        self.transition = Transition()

        # Reward / IRL components
        self.reward = reward.to(self.device)
        self.env = env
        self.feature_map = feature_map
        self.expert_storage: FeatureTrajectoryBuffer | None = None
        self.imitator_storage: FeatureTrajectoryBuffer | None = None

        # Shared hyperparameter
        self.gamma = self._validate_discount_gamma(float(gamma), name="gamma")
        if self.cfg.discount_gamma is None:
            self.irl_discount_gamma = self.gamma
        else:
            self.irl_discount_gamma = self._validate_discount_gamma(
                float(self.cfg.discount_gamma),
                name="cfg.discount_gamma",
            )
        self.normalize_returns_by_episode_length = bool(self.cfg.normalize_returns_by_episode_length)

        # Reward optimizer
        lr = self.cfg.reward_learning_rate
        if lr is None:
            lr = float(getattr(rl_alg, "learning_rate", 1e-4))

        self.reward_optimizer = optim.RMSprop(
            self.reward.parameters(),
            lr=lr,
            weight_decay=self.cfg.weight_decay,
        )

    @staticmethod
    def _validate_discount_gamma(gamma: float, *, name: str) -> float:
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"`{name}` must be in (0, 1], got {gamma}.")
        return float(gamma)

    # ---------------------------------------------------------------------
    # Storage initialization (using global RuntimeContext)
    # ---------------------------------------------------------------------
    def init_expert_storage(
        self,
        runtime_ctx: RuntimeContext,
        *,
        cfg: FeatureBufCfg | None = None,
        num_envs: int = 1,
    ) -> None:
        """
        Initialize expert feature storage.

        Uses the global RuntimeContext and overrides `num_envs` (default=1 for a
        single expert trajectory stream) while preserving feature_dim and any other
        runtime metadata.
        """
        buf_cfg = cfg or FeatureBufCfg(min_ep_len=1)
        expert_ctx = replace(runtime_ctx, num_envs=int(num_envs))
        self.expert_storage = FeatureTrajectoryBuffer(cfg=buf_cfg, ctx=expert_ctx, gamma=self.gamma)

    def init_imitator_storage(
        self,
        runtime_ctx: RuntimeContext,
        *,
        cfg: FeatureBufCfg | None = None,
    ) -> None:
        """
        Initialize imitator feature storage using the global RuntimeContext as-is.
        """
        buf_cfg = cfg or FeatureBufCfg(min_ep_len=1)
        self.imitator_storage = FeatureTrajectoryBuffer(cfg=buf_cfg, ctx=runtime_ctx, gamma=self.gamma)

    # ---------------------------------------------------------------------
    # Modes
    # ---------------------------------------------------------------------
    def eval_mode(self) -> None:
        if hasattr(self.rl_alg, "actor_critic"):
            self.rl_alg.actor_critic.eval()
        self.reward.eval()

    def train_mode(self) -> None:
        if hasattr(self.rl_alg, "actor_critic"):
            self.rl_alg.actor_critic.train()
        self.reward.train()

    # ---------------------------------------------------------------------
    # RL interaction / rollout collection
    # ---------------------------------------------------------------------
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute actions and (optionally) cache pre-step features from the live env.

        If `feature_map` + `env` are provided, we cache features here so they match
        the same timestep as the action (pre-step features).
        """
        if not hasattr(self.rl_alg, "actor_critic"):
            raise AttributeError("`rl_alg` must expose `actor_critic.act(obs)` for IRL.act().")

        # Cache pre-step features if env-based feature extraction is configured
        if self.feature_map is not None:
            if self.env is None:
                raise RuntimeError("`feature_map` is set but `env` is None.")
            with torch.no_grad():
                z = self.feature_map(self.env)
                if not isinstance(z, torch.Tensor):
                    z = torch.as_tensor(z)
                self.transition.features = z.detach()
        else:
            self.transition.features = None

        self.transition.actions = self.rl_alg.actor_critic.act(obs).detach()
        return self.transition.actions

    def process_env_step(self, dones: torch.Tensor, features: torch.Tensor | None = None) -> None:
        """
        Add a single vectorized timestep to the imitator feature buffer.

        Preferred usage:
          - call `act(obs)` first (caches pre-step features)
          - then call `process_env_step(dones)` after env.step(...)

        Alternatively, pass `features` explicitly.
        """
        if self.imitator_storage is None:
            raise RuntimeError("Imitator storage is not initialized. Call `init_imitator_storage(...)` first.")

        if features is None:
            features = self.transition.features

        if features is None:
            raise ValueError(
                "No features available. Either provide `features=` explicitly to "
                "`process_env_step(...)` or configure `env` + `feature_map` and call `act(obs)` first."
            )

        dones = rearrange(dones, "... -> (...)").to(torch.bool)
        self.imitator_storage.add_step(z=features, done=dones)

        # Clear cached timestep features after consuming them
        self.transition.features = None

    def add_expert_step(self, features: torch.Tensor, dones: torch.Tensor) -> None:
        if self.expert_storage is None:
            raise RuntimeError("Expert storage is not initialized. Call `init_expert_storage(...)` first.")
        self.expert_storage.add_step(z=features, done=rearrange(dones, "... -> (...)").to(torch.bool))

    def add_expert_episode(self, features: torch.Tensor) -> None:
        """
        Convenience helper for a single expert episode with one environment.
        `features` must be [T, D].
        """
        if self.expert_storage is None:
            raise RuntimeError("Expert storage is not initialized. Call `init_expert_storage(...)` first.")
        self.expert_storage.add_episode(features)

    def clear_imitator_storage(self) -> None:
        if self.imitator_storage is not None:
            self.imitator_storage.clear()

    # ---------------------------------------------------------------------
    # Reward evaluation on feature trajectories
    # ---------------------------------------------------------------------
    def _eval_expected_return(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Computes the mean discounted return over a batch of trajectories."""
        returns = FeatureTrajectoryBuffer.eval_discounted_returns_from_model(
            feats=feats,
            mask=mask,
            reward_model=self.reward,
            gamma=self.irl_discount_gamma,
        )
        if self.normalize_returns_by_episode_length:
            lengths = mask.to(device=returns.device, dtype=returns.dtype).sum(dim=1).clamp_min(1.0)
            returns = returns / lengths
        return returns.mean()

    # ---------------------------------------------------------------------
    # IRL update
    # ---------------------------------------------------------------------
    def reward_update(self) -> tuple[float, float]:
        if self.expert_storage is None:
            raise RuntimeError("Expert storage is not initialized.")
        if self.imitator_storage is None:
            raise RuntimeError("Imitator storage is not initialized.")
        if len(self.expert_storage) == 0:
            raise RuntimeError("Expert storage is empty.")
        if len(self.imitator_storage) == 0:
            raise RuntimeError("Imitator storage is empty.")

        total_reward_loss = 0.0
        total_grad_norm = 0.0
        num_updates = max(1, int(self.cfg.num_learning_epochs))
        batch_size = int(self.cfg.batch_size)

        for _ in range(num_updates):
            expert_feats, expert_mask, _ = self.expert_storage.sample_episodes(
                batch_size=batch_size, device=self.device
            )
            imitator_feats, imitator_mask, _ = self.imitator_storage.sample_episodes(
                batch_size=batch_size, device=self.device
            )
            current_returns = self._eval_expected_return(imitator_feats, imitator_mask)
            expert_returns = self._eval_expected_return(expert_feats, expert_mask)
            reward_loss = float(self.cfg.reward_loss_coef) * (current_returns - expert_returns)

            self.reward_optimizer.zero_grad(set_to_none=True)
            reward_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.reward.parameters(), float(self.cfg.max_grad_norm))
            self.reward_optimizer.step()

            total_reward_loss += float(reward_loss.item())
            total_grad_norm += float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

        return total_reward_loss / num_updates, total_grad_norm / num_updates
