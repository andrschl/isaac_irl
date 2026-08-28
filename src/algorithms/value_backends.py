"""Composable direct-Q and successor-feature action-value backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.off_policy_common import gradient_norm
from algorithms.scalar_action_value import (
    ScalarActionValue,
    ScalarActionValueEnsemble,
    polyak_update_scalar_value,
)
from algorithms.soft_value import entropy_return_td_target, soft_task_q_td_target
from algorithms.state_dependent_tanh_actor import TanhPolicyProtocol
from algorithms.successor_ensemble import (
    RunningStandardizer,
    SuccessorFeatureEnsemble,
    successor_ensemble_td_target,
    successor_return_scale_from_second_moment,
    update_ensemble_target,
)
from config.off_policy import DirectQCfg, OffPolicyTrainCfg, SuccessorQCfg
from reward_model.scorer import LinearRewardScorer, RewardScorer
from storage.transition_replay import TransitionReplayBuffer, TransitionRollout


@dataclass(frozen=True, slots=True)
class ValueEstimate:
    """Scalar action values and optional decomposition, each shaped [B]."""

    q: torch.Tensor
    reward_q: torch.Tensor
    entropy_q: torch.Tensor


class ActionValueBackend(Protocol):
    """Policy-independent interface implemented by both value representations."""

    num_heads: int

    def fit_feature_scale(self, rollout: TransitionRollout) -> None: ...

    def update(
        self,
        replay: TransitionReplayBuffer,
        *,
        target_policy: TanhPolicyProtocol,
        entropy_temperature: float,
    ) -> dict[str, float]: ...

    def mean(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        entropy_temperature: float,
    ) -> ValueEstimate: ...

    def modules(self) -> tuple[nn.Module, ...]: ...

    def save_state(self) -> dict[str, Any]: ...

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool) -> None: ...


class DirectActionValue:
    """Five-head scalar Q with paired targets and ensemble-mean control."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        reward: RewardScorer,
        observation_normalizer: RunningStandardizer,
        cfg: DirectQCfg,
        train_cfg: OffPolicyTrainCfg,
        device: torch.device,
        seed: int,
    ) -> None:
        self.num_heads = cfg.num_heads
        self.reward = reward
        self.observation_normalizer = observation_normalizer
        self.train_cfg = train_cfg
        self.device = device
        self.online = ScalarActionValueEnsemble(
            observation_dim,
            action_dim,
            num_heads=cfg.num_heads,
            hidden_dims=cfg.hidden_dims,
        ).to(device)
        self.target = self.online.target_copy()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=train_cfg.critic_learning_rate
        )
        self.replay_generators = [
            torch.Generator(device="cpu").manual_seed(seed + 1009 * (head + 1))
            for head in range(cfg.num_heads)
        ]
        self.action_generators = [
            torch.Generator(device=device).manual_seed(seed + 2003 * (head + 1))
            for head in range(cfg.num_heads)
        ]

    def fit_feature_scale(self, rollout: TransitionRollout) -> None:
        del rollout

    def update(
        self,
        replay: TransitionReplayBuffer,
        *,
        target_policy: TanhPolicyProtocol,
        entropy_temperature: float,
    ) -> dict[str, float]:
        """Apply one averaged optimizer step over independently sampled heads."""
        losses: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for head_index in range(self.num_heads):
            batch = replay.sample(
                self.train_cfg.critic_batch_size,
                generator=self.replay_generators[head_index],
                device=self.device,
            )
            with torch.no_grad():
                next_actions, next_log_probability = (
                    target_policy.sample_with_log_probability(
                        batch.next_policy_observations,
                        generator=self.action_generators[head_index],
                    )
                )
                next_q = self.target.head(
                    head_index,
                    self.observation_normalizer(batch.next_critic_observations),
                    next_actions,
                )
                rewards = self.reward.score(batch.features).to(next_q)
                target = soft_task_q_td_target(
                    rewards,
                    next_log_probability,
                    next_q,
                    batch.bootstrap_valid,
                    gamma=self.train_cfg.gamma,
                    entropy_coefficient=entropy_temperature,
                )
            prediction = self.online.head(
                head_index,
                self.observation_normalizer(batch.critic_observations),
                batch.actions,
            )
            losses.append(F.huber_loss(prediction, target))
            targets.append(target.mean())
        loss = torch.stack(losses).mean()
        _require_finite(loss, name="direct-Q TD loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        online_gradient_norm = gradient_norm(self.online.parameters())
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.train_cfg.max_grad_norm
        )
        self.optimizer.step()
        polyak_update_scalar_value(
            self.online, self.target, self.train_cfg.target_polyak
        )
        metrics = {
            "DirectQ/td_loss": float(loss.item()),
            "DirectQ/gradient_norm": online_gradient_norm,
            "DirectQ/target_mean": float(torch.stack(targets).mean().item()),
        }
        for head_index, head_loss in enumerate(losses):
            metrics[f"DirectQ/head_{head_index}_td_loss"] = float(
                head_loss.detach().item()
            )
        return metrics

    def mean(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        entropy_temperature: float,
    ) -> ValueEstimate:
        del entropy_temperature
        q = self.online.mean(self.observation_normalizer(observations), actions)
        zeros = torch.zeros_like(q)
        return ValueEstimate(q=q, reward_q=q, entropy_q=zeros)

    def modules(self) -> tuple[nn.Module, ...]:
        return self.online, self.target

    def save_state(self) -> dict[str, Any]:
        return {
            "type": "direct",
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_rng": [item.get_state() for item in self.replay_generators],
            "action_rng": [item.get_state() for item in self.action_generators],
        }

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool) -> None:
        if state.get("type") != "direct":
            raise ValueError("Checkpoint value backend is not direct Q.")
        self.online.load_state_dict(state["online"], strict=True)
        self.target.load_state_dict(state["target"], strict=True)
        for generator, generator_state in zip(
            self.replay_generators, state["replay_rng"], strict=True
        ):
            generator.set_state(generator_state.cpu())
        for generator, generator_state in zip(
            self.action_generators, state["action_rng"], strict=True
        ):
            generator.set_state(generator_state.cpu())
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])


class SuccessorActionValue:
    """Reward-independent vector Q plus an optional entropy-return critic."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        feature_dim: int,
        observation_normalizer: RunningStandardizer,
        reward: LinearRewardScorer,
        cfg: SuccessorQCfg,
        train_cfg: OffPolicyTrainCfg,
        use_entropy_critic: bool,
        device: torch.device,
        seed: int,
    ) -> None:
        if reward.feature_dim != feature_dim:
            raise ValueError("Reward and successor feature dimensions must match.")
        self.num_heads = cfg.num_heads
        self.reward = reward
        self.cfg = cfg
        self.observation_normalizer = observation_normalizer
        self.train_cfg = train_cfg
        self.device = device
        self.online = SuccessorFeatureEnsemble(
            observation_dim,
            action_dim,
            feature_dim,
            num_heads=cfg.num_heads,
            trunk_hidden_dims=cfg.trunk_hidden_dims,
            latent_dim=cfg.latent_dim,
            head_hidden_dim=cfg.head_hidden_dim,
        ).to(device)
        self.target = self.online.target_copy()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=train_cfg.critic_learning_rate
        )
        self.entropy_online = (
            ScalarActionValue(
                observation_dim,
                action_dim,
                hidden_dims=cfg.entropy_critic_hidden_dims,
            ).to(device)
            if use_entropy_critic
            else None
        )
        self.entropy_target = (
            self.entropy_online.target_copy()
            if self.entropy_online is not None
            else None
        )
        self.entropy_optimizer = (
            torch.optim.Adam(
                self.entropy_online.parameters(),
                lr=cfg.entropy_critic_learning_rate,
            )
            if self.entropy_online is not None
            else None
        )
        self.return_scale = torch.ones(feature_dim, device=device)
        self.scale_fitted = False
        self.scale_refits = 0
        self._feature_square_sum = torch.zeros(feature_dim, device=device)
        self._feature_count = 0
        self.replay_generators = [
            torch.Generator(device="cpu").manual_seed(seed + 1009 * (head + 1))
            for head in range(cfg.num_heads)
        ]
        self.action_generators = [
            torch.Generator(device=device).manual_seed(seed + 2003 * (head + 1))
            for head in range(cfg.num_heads)
        ]
        self.entropy_replay_generator = torch.Generator(device="cpu").manual_seed(
            seed + 47
        )
        self.entropy_action_generator = torch.Generator(device=device).manual_seed(
            seed + 53
        )

    @torch.no_grad()
    def fit_feature_scale(self, rollout: TransitionRollout) -> None:
        """Keep the successor output scale matched to the observed features.

        Features a competent policy produces (a lifted object, a reached goal)
        are identically zero in the first rollouts, so a scale fitted once there
        stays at its floor and those heads have to emit values orders of
        magnitude off their unit range. The running second moment tracks them,
        and a refit rescales the heads so predictions are unchanged.
        """
        features = rollout.features.flatten(0, 1).to(self.device)
        self._feature_square_sum += features.square().sum(dim=0)
        self._feature_count += features.shape[0]
        observed_scale = successor_return_scale_from_second_moment(
            self._feature_square_sum / self._feature_count,
            gamma=self.train_cfg.gamma,
            minimum_scale=self.cfg.return_scale_min,
        )
        if not self.scale_fitted:
            self._apply_return_scale(observed_scale, preserve_predictions=False)
            self.scale_fitted = True
            return
        if float((observed_scale / self.return_scale).max().item()) > (
            self.cfg.scale_refit_ratio
        ):
            self._apply_return_scale(observed_scale, preserve_predictions=True)
            self.scale_refits += 1

    @torch.no_grad()
    def _apply_return_scale(
        self, scale: torch.Tensor, *, preserve_predictions: bool
    ) -> None:
        """Set the scale on the online and target ensembles together."""
        self.online.set_output_scale(scale, preserve_predictions=preserve_predictions)
        self.target.set_output_scale(scale, preserve_predictions=preserve_predictions)
        self.return_scale = scale

    def update(
        self,
        replay: TransitionReplayBuffer,
        *,
        target_policy: TanhPolicyProtocol,
        entropy_temperature: float,
    ) -> dict[str, float]:
        if not self.scale_fitted:
            raise RuntimeError("Fit successor feature scales before TD updates.")
        losses: list[torch.Tensor] = []
        feature_losses: list[torch.Tensor] = []
        for head_index in range(self.num_heads):
            batch = replay.sample(
                self.train_cfg.critic_batch_size,
                generator=self.replay_generators[head_index],
                device=self.device,
            )
            with torch.no_grad():
                next_actions, _ = target_policy.sample_with_log_probability(
                    batch.next_policy_observations,
                    generator=self.action_generators[head_index],
                )
                next_successors = self.target.head(
                    head_index,
                    self.observation_normalizer(batch.next_critic_observations),
                    next_actions,
                )
                target = successor_ensemble_td_target(
                    batch.features,
                    next_successors,
                    batch.bootstrap_valid,
                    self.train_cfg.gamma,
                )
            prediction = self.online.head(
                head_index,
                self.observation_normalizer(batch.critic_observations),
                batch.actions,
            )
            residual = (prediction - target) / self.return_scale.unsqueeze(0)
            by_feature = F.huber_loss(
                residual, torch.zeros_like(residual), reduction="none"
            ).mean(dim=0)
            losses.append(by_feature.mean())
            feature_losses.append(by_feature.detach())
        loss = torch.stack(losses).mean()
        _require_finite(loss, name="successor TD loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        online_gradient_norm = gradient_norm(self.online.parameters())
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.train_cfg.max_grad_norm
        )
        self.optimizer.step()
        update_ensemble_target(self.online, self.target, self.train_cfg.target_polyak)
        metrics = {
            "SuccessorQ/td_loss": float(loss.item()),
            "SuccessorQ/gradient_norm": online_gradient_norm,
            "SuccessorQ/return_scale_min": float(self.return_scale.min().item()),
            "SuccessorQ/return_scale_max": float(self.return_scale.max().item()),
            "SuccessorQ/scale_refits": float(self.scale_refits),
        }
        per_head = torch.stack(feature_losses)
        for head_index in range(self.num_heads):
            metrics[f"SuccessorQ/head_{head_index}_td_loss"] = float(
                per_head[head_index].mean().item()
            )
        metrics.update(
            self._update_entropy(
                replay,
                target_policy=target_policy,
                entropy_temperature=entropy_temperature,
            )
        )
        return metrics

    def _update_entropy(
        self,
        replay: TransitionReplayBuffer,
        *,
        target_policy: TanhPolicyProtocol,
        entropy_temperature: float,
    ) -> dict[str, float]:
        if self.entropy_online is None:
            if entropy_temperature != 0.0:
                raise RuntimeError(
                    "Positive entropy temperature requires an entropy-return critic."
                )
            return {}
        if self.entropy_target is None or self.entropy_optimizer is None:
            raise RuntimeError("Incomplete entropy-return critic state.")
        batch = replay.sample(
            self.train_cfg.critic_batch_size,
            generator=self.entropy_replay_generator,
            device=self.device,
        )
        with torch.no_grad():
            next_actions, next_log_probability = (
                target_policy.sample_with_log_probability(
                    batch.next_policy_observations,
                    generator=self.entropy_action_generator,
                )
            )
            next_entropy_q = self.entropy_target(
                self.observation_normalizer(batch.next_critic_observations),
                next_actions,
            )
            target = entropy_return_td_target(
                next_log_probability,
                next_entropy_q,
                batch.bootstrap_valid,
                self.train_cfg.gamma,
            )
        prediction = self.entropy_online(
            self.observation_normalizer(batch.critic_observations),
            batch.actions,
        )
        loss = F.huber_loss(prediction, target)
        _require_finite(loss, name="entropy-return TD loss")
        self.entropy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        entropy_gradient_norm = gradient_norm(self.entropy_online.parameters())
        torch.nn.utils.clip_grad_norm_(
            self.entropy_online.parameters(), self.train_cfg.max_grad_norm
        )
        self.entropy_optimizer.step()
        polyak_update_scalar_value(
            self.entropy_online,
            self.entropy_target,
            self.train_cfg.target_polyak,
        )
        return {
            "EntropyQ/td_loss": float(loss.item()),
            "EntropyQ/gradient_norm": entropy_gradient_norm,
        }

    def mean(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        entropy_temperature: float,
    ) -> ValueEstimate:
        normalized_observations = self.observation_normalizer(observations)
        successors = self.online.mean(normalized_observations, actions)
        reward_weights = self.reward.weights.detach().to(successors)
        reward_q = self.reward.step_dt.to(successors) * torch.einsum(
            "bd,d->b", successors, reward_weights
        )
        if self.entropy_online is None:
            entropy_q = torch.zeros_like(reward_q)
        else:
            entropy_q = self.entropy_online(normalized_observations, actions)
        combined_q = reward_q + float(entropy_temperature) * entropy_q
        _require_finite(combined_q, name="successor scalarized Q")
        return ValueEstimate(
            q=combined_q,
            reward_q=reward_q,
            entropy_q=entropy_q,
        )

    def modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [self.online, self.target]
        if self.entropy_online is not None:
            if self.entropy_target is None:
                raise RuntimeError("Missing entropy target.")
            modules.extend((self.entropy_online, self.entropy_target))
        return tuple(modules)

    def save_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "type": "successor",
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "return_scale": self.return_scale.detach().cpu(),
            "scale_fitted": self.scale_fitted,
            "scale_refits": self.scale_refits,
            "feature_square_sum": self._feature_square_sum.detach().cpu(),
            "feature_count": self._feature_count,
            "replay_rng": [item.get_state() for item in self.replay_generators],
            "action_rng": [item.get_state() for item in self.action_generators],
            "entropy_replay_rng": self.entropy_replay_generator.get_state(),
            "entropy_action_rng": self.entropy_action_generator.get_state(),
        }
        if self.entropy_online is not None:
            if self.entropy_target is None or self.entropy_optimizer is None:
                raise RuntimeError("Incomplete entropy critic state.")
            state.update(
                {
                    "entropy_online": self.entropy_online.state_dict(),
                    "entropy_target": self.entropy_target.state_dict(),
                    "entropy_optimizer": self.entropy_optimizer.state_dict(),
                }
            )
        return state

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool) -> None:
        if state.get("type") != "successor":
            raise ValueError("Checkpoint value backend is not successor Q.")
        checkpoint_has_entropy = "entropy_online" in state
        if checkpoint_has_entropy != (self.entropy_online is not None):
            raise ValueError("Checkpoint entropy-critic configuration differs.")
        self.online.load_state_dict(state["online"], strict=True)
        self.target.load_state_dict(state["target"], strict=True)
        self.return_scale = state["return_scale"].to(self.return_scale)
        self.online.set_output_scale(self.return_scale)
        self.target.set_output_scale(self.return_scale)
        self.scale_fitted = bool(state["scale_fitted"])
        self.scale_refits = int(state["scale_refits"])
        self._feature_square_sum = state["feature_square_sum"].to(
            self._feature_square_sum
        )
        self._feature_count = int(state["feature_count"])
        for generator, generator_state in zip(
            self.replay_generators, state["replay_rng"], strict=True
        ):
            generator.set_state(generator_state.cpu())
        for generator, generator_state in zip(
            self.action_generators, state["action_rng"], strict=True
        ):
            generator.set_state(generator_state.cpu())
        self.entropy_replay_generator.set_state(state["entropy_replay_rng"].cpu())
        self.entropy_action_generator.set_state(state["entropy_action_rng"].cpu())
        if self.entropy_online is not None:
            if self.entropy_target is None or self.entropy_optimizer is None:
                raise RuntimeError("Incomplete entropy critic state.")
            self.entropy_online.load_state_dict(state["entropy_online"], strict=True)
            self.entropy_target.load_state_dict(state["entropy_target"], strict=True)
            if load_optimizer:
                self.entropy_optimizer.load_state_dict(state["entropy_optimizer"])
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
