from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

from interfaces import FeatureSpec, FeatureStep
from reward_model.base import BaseRewardModel
from storage.feature_storage import FeatureBufCfg, FeatureTrajectoryBuffer
from utils.runtime_context import RuntimeContext


@dataclass(slots=True)
class IRLCfg:
    """IRL-specific reward-learning hyperparameters/config."""

    expert_data_path: str = ""
    expert_num_trajectories: int | None = (
        None  # None = use all; int = max trajectories for ablations
    )
    expert_subset_strategy: str = "first"  # "first" | "random"
    batch_size: int = 256
    num_learning_epochs: int = 1
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0
    reward_loss_coef: float = 1.0
    reward_learning_rate: float | None = None
    discount_gamma: float | None = None
    normalize_returns_by_episode_length: bool = True


class IrlAlgorithm:
    """Feature-buffer reward learner.

    Owns:
      - a learned reward model that consumes feature trajectories
        ``(feats [B, T, D], mask [B, T])``,
      - two :class:`FeatureTrajectoryBuffer` instances (expert + imitator),
      - the optimizer that fits the reward model to the
        ``E[expert returns] - E[imitator returns]`` objective.

    Buffers are built at construction; the caller is responsible for rolling
    out the policy, feeding imitator feature steps via :meth:`add_imitator`,
    populating the expert buffer via :meth:`add_expert_episode`, and
    triggering :meth:`update`.
    """

    def __init__(
        self,
        reward: BaseRewardModel,
        *,
        feature_spec: FeatureSpec,
        num_envs: int,
        gamma: float,
        cfg: IRLCfg | None = None,
        device: str | torch.device = "cpu",
        imitator_buffer_cfg: FeatureBufCfg | None = None,
        expert_buffer_cfg: FeatureBufCfg | None = None,
        expert_num_envs: int = 1,
        expert_success_rate: float | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.cfg = cfg or IRLCfg()
        self._validate_cfg(self.cfg)

        self.reward_model = reward.to(self.device)
        self.feature_spec = feature_spec
        self.feature_names = (
            list(feature_spec.names) if feature_spec.names is not None else None
        )
        self.expert_success_rate = (
            float(expert_success_rate) if expert_success_rate is not None else None
        )

        self.gamma = self._validate_discount_gamma(float(gamma), name="gamma")
        if self.cfg.discount_gamma is None:
            self.irl_discount_gamma = self.gamma
        else:
            self.irl_discount_gamma = self._validate_discount_gamma(
                float(self.cfg.discount_gamma),
                name="cfg.discount_gamma",
            )
        self.normalize_returns_by_episode_length = bool(
            self.cfg.normalize_returns_by_episode_length
        )

        if num_envs <= 0:
            raise ValueError(f"`num_envs` must be > 0, got {num_envs}.")
        if expert_num_envs <= 0:
            raise ValueError(f"`expert_num_envs` must be > 0, got {expert_num_envs}.")

        imitator_ctx = RuntimeContext(
            num_envs=int(num_envs),
            feature_dim=int(feature_spec.dim),
            device=str(self.device),
        )
        expert_ctx = RuntimeContext(
            num_envs=int(expert_num_envs),
            feature_dim=int(feature_spec.dim),
            device=str(self.device),
        )
        self.imitator_storage = FeatureTrajectoryBuffer(
            cfg=imitator_buffer_cfg or FeatureBufCfg(min_ep_len=1),
            ctx=imitator_ctx,
            gamma=self.gamma,
        )
        self.expert_storage = FeatureTrajectoryBuffer(
            cfg=expert_buffer_cfg or FeatureBufCfg(min_ep_len=1, store_device="cpu"),
            ctx=expert_ctx,
            gamma=self.gamma,
        )

        lr = (
            1e-4
            if self.cfg.reward_learning_rate is None
            else float(self.cfg.reward_learning_rate)
        )

        self.reward_optimizer = optim.RMSprop(
            self.reward_model.parameters(),
            lr=lr,
            weight_decay=float(self.cfg.weight_decay),
        )

    @staticmethod
    def _validate_cfg(cfg: IRLCfg) -> None:
        if (
            cfg.expert_num_trajectories is not None
            and int(cfg.expert_num_trajectories) <= 0
        ):
            raise ValueError(
                "`expert_num_trajectories` must be > 0 or None, "
                f"got {cfg.expert_num_trajectories}."
            )
        if cfg.expert_subset_strategy not in {"first", "random"}:
            raise ValueError(
                "`expert_subset_strategy` must be 'first' or 'random', "
                f"got {cfg.expert_subset_strategy!r}."
            )
        if cfg.batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {cfg.batch_size}.")
        if cfg.num_learning_epochs <= 0:
            raise ValueError(
                f"`num_learning_epochs` must be > 0, got {cfg.num_learning_epochs}."
            )
        if cfg.weight_decay < 0.0:
            raise ValueError(f"`weight_decay` must be >= 0, got {cfg.weight_decay}.")
        if cfg.max_grad_norm <= 0.0:
            raise ValueError(f"`max_grad_norm` must be > 0, got {cfg.max_grad_norm}.")
        if cfg.reward_loss_coef < 0.0:
            raise ValueError(
                f"`reward_loss_coef` must be >= 0, got {cfg.reward_loss_coef}."
            )
        if (
            cfg.reward_learning_rate is not None
            and float(cfg.reward_learning_rate) <= 0.0
        ):
            raise ValueError(
                "`reward_learning_rate` must be > 0 or None, "
                f"got {cfg.reward_learning_rate}."
            )

    @staticmethod
    def _validate_discount_gamma(gamma: float, *, name: str) -> float:
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"`{name}` must be in (0, 1], got {gamma}.")
        return float(gamma)

    def add_imitator(self, step: FeatureStep) -> None:
        """Append one vectorized timestep to the imitator feature buffer.

        Shapes:
            step.features: [N, D]
            step.dones:    [N]
        """
        self.imitator_storage.add_step(z=step.features, done=step.dones.to(torch.bool))

    def add_expert_episode(self, features: torch.Tensor) -> None:
        """Append one complete expert episode. Shape: ``features [T, D]``. Single-env buffer only."""
        self.expert_storage.add_episode(features)

    def replace_imitator_trajectories(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Replace current-policy data from one grouped collection.

        Args:
            features: [C,G,T,D] raw reward features
            valid: [C,G,T] bool first-episode mask
        """
        if features.ndim != 4:
            raise ValueError(
                f"Expected features [C,G,T,D], got {tuple(features.shape)}."
            )
        if tuple(valid.shape) != tuple(features.shape[:3]) or valid.dtype != torch.bool:
            raise ValueError(
                f"Expected bool valid shape {tuple(features.shape[:3])}, got "
                f"{tuple(valid.shape)} with dtype {valid.dtype}."
            )
        if features.shape[-1] != self.feature_spec.dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_spec.dim}, got {features.shape[-1]}."
            )
        batch_features = rearrange(features, "c g t d -> (c g) t d")
        batch_valid = rearrange(valid, "c g t -> (c g) t")
        self.imitator_storage.replace_episode_batch(batch_features, batch_valid)

    def clear_imitator(self) -> None:
        self.imitator_storage.clear()

    def finalize_imitator(self) -> None:
        self.imitator_storage.finalize_in_progress_episodes()

    @property
    def batch_size(self) -> int:
        return int(self.cfg.batch_size)

    def can_update(self) -> bool:
        return len(self.expert_storage) > 0 and len(self.imitator_storage) > 0

    def per_feature_return_mean(
        self, feats: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Per-feature mean discounted return over a batch.

        Shapes:
            feats: [B, T, D]
            mask:  [B, T]
        Returns:
            [D]  - ``E_traj[sum_t gamma^t f_t]`` averaged over the batch, with
            optional per-episode length normalization applied before the
            outer mean.
        """
        discounted_feats = FeatureTrajectoryBuffer.discounted_feature_returns(
            feats,
            mask,
            gamma=self.irl_discount_gamma,
        )  # [B, D]
        if self.normalize_returns_by_episode_length:
            lengths = mask.to(dtype=discounted_feats.dtype).sum(dim=1).clamp_min(1.0)
            discounted_feats = discounted_feats / rearrange(lengths, "b -> b 1")
        return discounted_feats.mean(dim=0).detach()

    def sample_expert_feature_return_mean(
        self,
        device: torch.device | str,
    ) -> torch.Tensor | None:
        """Sample expert episodes and return per-feature mean returns [D]."""
        if len(self.expert_storage) == 0:
            return None
        feats, mask, _ = self.expert_storage.sample_episodes(
            batch_size=self.batch_size, device=device
        )
        return self.per_feature_return_mean(feats, mask)

    def eval_expected_return(
        self, feats: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean discounted return + per-feature mean return over a batch.

        Shapes:
            feats: [B, T, D]
            mask:  [B, T]
        Returns:
            scalar_mean_return: []
            per_feature_mean:   [D]
        """
        scalar_returns = self.reward_model.discounted_returns_from_features(
            feats=feats,
            mask=mask,
            gamma=self.irl_discount_gamma,
        )  # [B]

        if self.normalize_returns_by_episode_length:
            lengths = (
                mask.to(device=scalar_returns.device, dtype=scalar_returns.dtype)
                .sum(dim=1)
                .clamp_min(1.0)
            )  # [B]
            scalar_returns = scalar_returns / lengths

        return scalar_returns.mean(), self.per_feature_return_mean(feats, mask)

    def update(self) -> dict[str, float]:
        """Run reward updates and return scalar metrics.

        ``feature_exp_diff_norm = ||E[mu_E] - E[mu_pi]||`` where ``mu`` are the
        per-feature mean discounted returns averaged across minibatches.
        """
        if len(self.expert_storage) == 0:
            raise RuntimeError("Expert storage is empty.")
        if len(self.imitator_storage) == 0:
            raise RuntimeError("Imitator storage is empty.")

        num_updates = int(self.cfg.num_learning_epochs)
        batch_size = int(self.cfg.batch_size)
        reward_losses: list[float] = []
        expert_per_feats: list[torch.Tensor] = []
        imitator_per_feats: list[torch.Tensor] = []

        for _ in range(num_updates):
            expert_feats, expert_mask, _ = self.expert_storage.sample_episodes(
                batch_size=batch_size, device=self.device
            )
            imitator_feats, imitator_mask, _ = self.imitator_storage.sample_episodes(
                batch_size=batch_size, device=self.device
            )
            current_returns, imitator_per_feat = self.eval_expected_return(
                imitator_feats, imitator_mask
            )
            expert_returns, expert_per_feat = self.eval_expected_return(
                expert_feats, expert_mask
            )
            reward_loss = float(self.cfg.reward_loss_coef) * (
                current_returns - expert_returns
            )

            loss = reward_loss + self.reward_model.get_regularization_loss()
            self.reward_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.reward_model.parameters(), float(self.cfg.max_grad_norm)
            )
            self.reward_optimizer.step()

            self.reward_model.apply_proximal_step()
            self.reward_model.project_weights()

            reward_losses.append(float(reward_loss.item()))
            expert_per_feats.append(expert_per_feat)
            imitator_per_feats.append(imitator_per_feat)

        expert_mean = torch.stack(expert_per_feats).mean(dim=0)
        imitator_mean = torch.stack(imitator_per_feats).mean(dim=0)
        feat_diff_norm = float((expert_mean - imitator_mean).norm().item())

        return {
            "IRL/reward_loss": sum(reward_losses) / len(reward_losses),
            "IRL/feature_exp_diff_norm": feat_diff_norm,
        }

    def train_metrics(self) -> dict[str, float]:
        """L2 norm and max-abs of reward model parameters."""
        total_sq = 0.0
        max_abs = 0.0
        with torch.no_grad():
            for param in self.reward_model.parameters():
                total_sq += float(param.detach().pow(2).sum().item())
                max_abs = max(max_abs, float(param.detach().abs().max().item()))
        return {
            "IRL/reward_param_norm": total_sq**0.5,
            "IRL/reward_param_max_abs": max_abs,
        }

    def train_mode(self) -> None:
        self.reward_model.train()

    def eval_mode(self) -> None:
        self.reward_model.eval()

    def save_state(self) -> dict[str, Any]:
        return {
            "reward_model_state_dict": self.reward_model.state_dict(),
            "reward_optimizer_state_dict": self.reward_optimizer.state_dict(),
        }

    def load_state(
        self, checkpoint: dict[str, Any], *, load_optimizer: bool = True
    ) -> None:
        reward_state = checkpoint.get("reward_model_state_dict")
        if reward_state is not None:
            self.reward_model.load_state_dict(reward_state)

        reward_optim_state = checkpoint.get("reward_optimizer_state_dict")
        if load_optimizer and reward_optim_state is not None:
            self.reward_optimizer.load_state_dict(reward_optim_state)
