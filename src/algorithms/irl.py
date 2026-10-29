from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange

from config.schema import FeatureBufCfg, IRLCfg
from interfaces import FeatureSpec
from reward.base import BaseRewardModel
from reward.model import LinearFeatureRewardModel
from storage.feature_storage import FeatureTrajectoryBuffer


class IrlAlgorithm:
    """Feature-buffer reward learner.

    Owns:
      - a learned reward model that consumes feature trajectories
        ``(feats [B, T, D], mask [B, T])``,
      - two :class:`FeatureTrajectoryBuffer` instances (expert + imitator),
      - the optimizer that fits the reward model to the
        ``E[expert returns] - E[imitator returns]`` objective.

    Buffers are built at construction; the caller is responsible for rolling
    out the policy, handing its feature trajectories to
    :meth:`replace_imitator_trajectories`, filling the expert buffer via
    :meth:`add_expert_episode`, and
    triggering :meth:`update`.
    """

    def __init__(
        self,
        reward: BaseRewardModel,
        *,
        feature_spec: FeatureSpec,
        gamma: float,
        step_dt: float = 1.0,
        cfg: IRLCfg | None = None,
        device: str | torch.device = "cpu",
        imitator_buffer_cfg: FeatureBufCfg | None = None,
        expert_buffer_cfg: FeatureBufCfg | None = None,
        expert_success_rate: float | None = None,
        environment_reward_weights: torch.Tensor | None = None,
    ) -> None:
        self.device = torch.device(device)
        if step_dt <= 0:
            raise ValueError("step_dt must be positive.")
        self.step_dt = float(step_dt)
        self.cfg = cfg or IRLCfg()

        self.reward_model = reward.to(self.device)
        self.feature_spec = feature_spec
        self.feature_names = (
            list(feature_spec.names) if feature_spec.names is not None else None
        )
        self.expert_success_rate = (
            float(expert_success_rate) if expert_success_rate is not None else None
        )
        # The environment's own reward weights, when the task exposes them, so
        # the run can report how close the learned reward is to the one it is
        # trying to recover. Never used as a training signal.
        self.environment_reward_weights = (
            None
            if environment_reward_weights is None
            else environment_reward_weights.detach().to(self.device)
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

        self.imitator_storage = self._build_buffer(
            imitator_buffer_cfg or FeatureBufCfg(min_ep_len=1)
        )
        self.expert_storage = self._build_buffer(
            expert_buffer_cfg or FeatureBufCfg(min_ep_len=1, store_device="cpu")
        )

        if self.cfg.solver == "exact_linear":
            if not isinstance(self.reward_model, LinearFeatureRewardModel):
                raise ValueError(
                    "`irl.solver=exact_linear` requires a linear reward model."
                )
            if self.reward_model.cfg.linear_projection != "l2_ball":
                raise ValueError(
                    "`irl.solver=exact_linear` requires an L2-ball projection."
                )
            self.reward_optimizer = None
        else:
            self.reward_optimizer = optim.RMSprop(
                self.reward_model.parameters(),
                lr=float(self.cfg.reward_learning_rate),
                weight_decay=float(self.cfg.weight_decay),
            )

    def _build_buffer(self, cfg: FeatureBufCfg) -> FeatureTrajectoryBuffer:
        return FeatureTrajectoryBuffer(
            cfg,
            feature_dim=int(self.feature_spec.dim),
            gamma=self.gamma,
            device=self.device,
        )

    @staticmethod
    def _validate_discount_gamma(gamma: float, *, name: str) -> float:
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"`{name}` must be in (0, 1], got {gamma}.")
        return float(gamma)

    def add_expert_episode(self, features: torch.Tensor) -> None:
        """Store one complete expert episode. Shapes: features [T, D]."""
        self.expert_storage.append_episode(features)

    def replace_imitator_episodes(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Install complete current-policy episodes: features [B,T,D], mask [B,T]."""
        self.imitator_storage.replace_episode_batch(features, mask)

    def replace_imitator_trajectories(
        self,
        features: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Replace current-policy data with one freshly collected generation.

        ``dones`` says where episodes ended, so the discounted feature returns
        compared against the expert's are accumulated over episodes rather than
        over an arbitrary collection window.

        Args:
            features: [N,T,D] raw reward features, one row per environment
            dones: [N,T] bool, true on the last step of an episode
        """
        if features.ndim != 3:
            raise ValueError(f"Expected features [N,T,D], got {tuple(features.shape)}.")
        if features.shape[-1] != self.feature_spec.dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_spec.dim}, got "
                f"{features.shape[-1]}."
            )
        self.imitator_storage.clear()
        self.imitator_storage.add_rollout(features, dones)

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
        return self.step_dt * discounted_feats.mean(dim=0).detach()

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

        return self.step_dt * scalar_returns.mean(), self.per_feature_return_mean(
            feats, mask
        )

    def update(self) -> dict[str, float]:
        """Run reward updates and return scalar metrics.

        ``feature_exp_diff_norm = ||E[mu_E] - E[mu_pi]||`` where ``mu`` are the
        per-feature mean discounted returns averaged across minibatches.
        """
        if len(self.expert_storage) == 0:
            raise RuntimeError("Expert storage is empty.")
        if len(self.imitator_storage) == 0:
            raise RuntimeError("Imitator storage is empty.")
        if self.cfg.solver == "exact_linear":
            return self._update_exact_linear()

        gradient_steps = int(self.cfg.gradient_steps)
        batch_size = int(self.cfg.batch_size)
        reward_losses: list[float] = []
        expert_per_feats: list[torch.Tensor] = []
        imitator_per_feats: list[torch.Tensor] = []

        for _ in range(gradient_steps):
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
            if self.reward_optimizer is None:
                raise RuntimeError("Gradient reward update has no optimizer.")
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
            "IRL/reward_optimizer_steps": float(gradient_steps),
        }

    def _update_exact_linear(self) -> dict[str, float]:
        """Solve the regularized linear dual on every stored episode exactly."""
        model = self.reward_model
        if not isinstance(model, LinearFeatureRewardModel):
            raise RuntimeError("Exact linear update requires a linear reward model.")
        mean_kwargs = {
            "gamma": self.irl_discount_gamma,
            "device": self.device,
            "normalize_by_episode_length": self.normalize_returns_by_episode_length,
        }
        expert_raw = self.step_dt * self.expert_storage.mean_feature_return(
            **mean_kwargs
        )
        imitator_raw = self.step_dt * self.imitator_storage.mean_feature_return(
            **mean_kwargs
        )
        scales = model.feature_scales.to(expert_raw)
        expert = expert_raw / scales
        imitator = imitator_raw / scales
        gap = expert - imitator
        radius = float(model.cfg.linear_projection_radius)
        regularization = (
            float(model.cfg.regularization_strength)
            if model.cfg.regularization == "l2"
            else 0.0
        )
        coefficient = float(self.cfg.reward_loss_coef)
        gap_norm = float(gap.norm().item())
        if gap_norm == 0.0 or coefficient == 0.0:
            best_response = torch.zeros_like(gap)
        elif regularization > 0.0:
            best_response = coefficient * gap / (2.0 * regularization)
            best_response = best_response * min(
                1.0, radius / float(best_response.norm().item())
            )
        else:
            best_response = radius * gap / gap_norm
        with torch.no_grad():
            model.reward.weight.copy_(best_response.unsqueeze(0))
        reward_loss = coefficient * torch.dot(best_response, imitator - expert)
        result = {
            "IRL/reward_loss": float(reward_loss.item()),
            "IRL/reward_regularization": regularization
            * float(best_response.square().sum().item()),
            "IRL/feature_exp_diff_norm": gap_norm,
            "IRL/raw_feature_exp_diff_norm": float(
                (expert_raw - imitator_raw).norm().item()
            ),
            "IRL/reward_optimizer_steps": 0.0,
            "IRL/reward_exact_solves": 1.0,
        }
        if self.feature_names is not None:
            for column, name in enumerate(self.feature_names):
                result[f"IRL/feature_gap/{name}"] = float(gap[column].item())
                result[f"IRL/expert_feature_exp/{name}"] = float(expert[column].item())
                result[f"IRL/imitator_feature_exp/{name}"] = float(
                    imitator[column].item()
                )
                result[f"IRL/weight/{name}"] = float(best_response[column].item())
                result[f"IRL/effective_weight/{name}"] = float(
                    model.effective_weights[column].item()
                )
                result[f"IRL/reward_contribution/{name}"] = float(
                    (best_response[column] * imitator[column]).item()
                )
        return result

    def train_metrics(self) -> dict[str, float]:
        """L2 norm and max-abs of reward model parameters."""
        total_sq = 0.0
        max_abs = 0.0
        with torch.no_grad():
            for param in self.reward_model.parameters():
                total_sq += float(param.detach().pow(2).sum().item())
                max_abs = max(max_abs, float(param.detach().abs().max().item()))
        metrics = {
            "IRL/reward_param_norm": total_sq**0.5,
            "IRL/reward_param_max_abs": max_abs,
        }
        cosine = self.reward_weight_cosine()
        if cosine is not None:
            metrics["IRL/reward_weight_cosine"] = cosine
        return metrics

    def reward_weight_cosine(self) -> float | None:
        """Cosine between the learned linear reward and the environment's own.

        The most direct read on whether IRL is recovering the task's reward
        rather than some other one that happens to fit the demonstrations.
        ``None`` when the reward is not linear or the task exposes no weights.
        """
        weights = getattr(self.reward_model, "effective_weights", None)
        if weights is None:
            weights = getattr(
                getattr(self.reward_model, "reward", None), "weight", None
            )
        if weights is None or self.environment_reward_weights is None:
            return None
        learned = weights.detach().reshape(-1)
        reference = self.environment_reward_weights.to(learned)
        if float(learned.norm()) == 0.0 or float(reference.norm()) == 0.0:
            return None
        return float(
            torch.nn.functional.cosine_similarity(learned, reference, dim=0).item()
        )

    def train_mode(self) -> None:
        self.reward_model.train()

    def eval_mode(self) -> None:
        self.reward_model.eval()

    def save_state(self) -> dict[str, Any]:
        state = {
            "reward_model_state_dict": self.reward_model.state_dict(),
            "reward_solver": self.cfg.solver,
            "feature_names": self.feature_names,
        }
        if isinstance(self.reward_model, LinearFeatureRewardModel):
            state["feature_scales"] = self.reward_model.feature_scales.detach().cpu()
        if self.reward_optimizer is not None:
            state["reward_optimizer_state_dict"] = self.reward_optimizer.state_dict()
        return state

    def load_state(
        self, checkpoint: dict[str, Any], *, load_optimizer: bool = True
    ) -> None:
        """Restore the reward model, failing rather than resuming untrained."""
        missing = {"reward_model_state_dict"} - set(checkpoint)
        if missing:
            raise ValueError(
                f"Reward-learner checkpoint is missing {sorted(missing)}; it has "
                f"{sorted(checkpoint)}."
            )
        saved_solver = checkpoint.get("reward_solver", "gradient")
        if saved_solver != self.cfg.solver:
            raise ValueError(
                f"Reward checkpoint solver {saved_solver!r} does not match "
                f"configured solver {self.cfg.solver!r}."
            )
        if (
            "feature_names" in checkpoint
            and checkpoint["feature_names"] != self.feature_names
        ):
            raise ValueError(
                "Reward checkpoint feature names do not match this run's learned basis."
            )
        if "feature_scales" in checkpoint:
            if not isinstance(
                self.reward_model, LinearFeatureRewardModel
            ) or not torch.equal(
                torch.as_tensor(checkpoint["feature_scales"], device="cpu"),
                self.reward_model.feature_scales.cpu(),
            ):
                raise ValueError(
                    "Reward checkpoint feature scales do not match this run's learned basis."
                )
        self.reward_model.load_state_dict(checkpoint["reward_model_state_dict"])
        if load_optimizer and self.reward_optimizer is not None:
            self.reward_optimizer.load_state_dict(
                checkpoint["reward_optimizer_state_dict"]
            )
