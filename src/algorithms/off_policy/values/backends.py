"""Composable direct-Q and successor-feature action-value backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from algorithms.off_policy.actor import TanhPolicyProtocol
from algorithms.off_policy.common import gradient_norm
from algorithms.off_policy.values.networks import (
    ScalarActionValue,
    ScalarActionValueEnsemble,
    SuccessorFeatureEnsemble,
    successor_ensemble_td_target,
    successor_return_scale_from_second_moment,
)
from algorithms.off_policy.values.parameter_ops import polyak_update
from algorithms.off_policy.values.targets import (
    entropy_return_td_target,
    soft_task_q_td_target,
)
from config.schema import DirectQCfg, OffPolicyTrainCfg, SuccessorQCfg
from reward.scorer import LinearRewardScorer, RewardScorer
from storage.transition_replay import (
    TransitionBatch,
    TransitionReplayBuffer,
    TransitionRollout,
)


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
        entropy_rate: float,
    ) -> dict[str, float]: ...

    def mean(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        entropy_rate: float,
    ) -> ValueEstimate: ...

    def modules(self) -> tuple[nn.Module, ...]: ...

    def reset_parameters(self) -> None: ...

    def save_state(self) -> dict[str, Any]: ...

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool) -> None: ...


def _optimize(
    loss: torch.Tensor,
    *,
    parameters: Any,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
    name: str,
) -> float:
    """Backward, clip, and step one loss; return the pre-clip gradient norm."""
    _require_finite(loss, name=name)
    parameter_list = list(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = gradient_norm(parameter_list)
    torch.nn.utils.clip_grad_norm_(parameter_list, max_grad_norm)
    optimizer.step()
    return norm


def head_rows(head_index: int, batch_size: int) -> slice:
    """The slice of one draw that a single ensemble head learns from."""
    return slice(head_index * batch_size, (head_index + 1) * batch_size)


class EntropyReturnCritic:
    """The scalar critic that carries a successor run's entropy return.

    Successor features are reward-independent by construction, so they cannot
    represent the entropy bonus a maximum-entropy objective adds: that term
    depends on the policy, not on the environment's features. It gets its own
    scalar critic, learning

        target_H = gamma * mask * (-dt log pi(a'|s') + target_Q_H(s',a')),

    and control uses ``Q_reward + entropy_rate * Q_H``. It exists only when the
    entropy rate is positive, so the whole object is absent rather than
    a group of fields that are each individually None.
    """

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        cfg: SuccessorQCfg,
        train_cfg: OffPolicyTrainCfg,
        step_dt: float,
        device: torch.device,
        seed: int,
    ) -> None:
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be > 0, got {step_dt}.")
        self.train_cfg = train_cfg
        self.step_dt = float(step_dt)
        self.learning_rate = cfg.entropy_critic_learning_rate
        self.device = device
        self.online = ScalarActionValue(
            observation_dim,
            action_dim,
            width=cfg.width,
            residual_blocks=cfg.residual_blocks,
        ).to(device)
        self.target = self.online.target_copy()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.learning_rate
        )
        self.replay_generator = torch.Generator(device="cpu").manual_seed(seed + 47)
        self.action_generator = torch.Generator(device=device).manual_seed(seed + 53)

    def value(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Entropy return of a batch of actions. Shapes: [B,O], [B,A] -> [B]."""
        return self.online(observations, actions)

    def update(
        self, replay: TransitionReplayBuffer, *, target_policy: TanhPolicyProtocol
    ) -> dict[str, float]:
        """Take one TD step on the entropy return."""
        batch = replay.sample(
            self.train_cfg.critic_batch_size,
            generator=self.replay_generator,
            device=self.device,
        )
        with torch.no_grad():
            next_actions, next_log_probability = (
                target_policy.sample_with_log_probability(
                    batch.next_policy_observations, generator=self.action_generator
                )
            )
            target = entropy_return_td_target(
                next_log_probability,
                self.target(batch.next_critic_observations, next_actions),
                batch.bootstrap_valid,
                gamma=self.train_cfg.gamma,
                step_dt=self.step_dt,
            )
        loss = F.huber_loss(
            self.online(batch.critic_observations, batch.actions), target
        )
        gradient_norm_value = _optimize(
            loss,
            parameters=self.online.parameters(),
            optimizer=self.optimizer,
            max_grad_norm=self.train_cfg.max_grad_norm,
            name="entropy-return TD loss",
        )
        polyak_update(self.online, self.target, self.train_cfg.target_polyak)
        return {
            "EntropyQ/td_loss": float(loss.item()),
            "EntropyQ/gradient_norm": gradient_norm_value,
        }

    def reset_parameters(self) -> None:
        self.online.reset_parameters()
        self.target = self.online.target_copy()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.learning_rate
        )

    def modules(self) -> tuple[nn.Module, ...]:
        return self.online, self.target

    def save_state(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_rng": self.replay_generator.get_state(),
            "action_rng": self.action_generator.get_state(),
        }

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool) -> None:
        self.online.load_state_dict(state["online"], strict=True)
        self.target.load_state_dict(state["target"], strict=True)
        self.replay_generator.set_state(state["replay_rng"].cpu())
        self.action_generator.set_state(state["action_rng"].cpu())
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])


class DirectActionValue:
    """Scalar Q ensemble with clipped bootstrap targets and pessimistic control."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        reward: RewardScorer,
        cfg: DirectQCfg,
        train_cfg: OffPolicyTrainCfg,
        step_dt: float,
        device: torch.device,
        seed: int,
    ) -> None:
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be > 0, got {step_dt}.")
        self.num_heads = cfg.num_heads
        self.control_reduction = cfg.control_reduction
        self.target_reduction = cfg.target_reduction
        self.target_subset_size = cfg.target_subset_size
        self.reward = reward
        self.train_cfg = train_cfg
        self.step_dt = float(step_dt)
        self.device = device
        self.online = ScalarActionValueEnsemble(
            observation_dim,
            action_dim,
            num_heads=cfg.num_heads,
            width=cfg.width,
            residual_blocks=cfg.residual_blocks,
        ).to(device)
        self.target = self.online.target_copy()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=train_cfg.critic_learning_rate
        )
        self.replay_generator = torch.Generator(device="cpu").manual_seed(seed + 1009)
        self.action_generator = torch.Generator(device=device).manual_seed(seed + 2003)
        self.subset_generator = torch.Generator(device="cpu").manual_seed(seed + 3001)

    def fit_feature_scale(self, rollout: TransitionRollout) -> None:
        del rollout

    def _bootstrap_value(self, per_head: torch.Tensor, head_index: int) -> torch.Tensor:
        """Reduce target values [M,B] to the one head ``head_index`` bootstraps from.

        A fresh subset each update is what stops the clip from concentrating on
        one pair of heads; drawing it on CPU keeps it reproducible from the
        checkpointed generator state.
        """
        if self.target_reduction == "independent":
            return per_head[head_index]
        subset = torch.randperm(self.num_heads, generator=self.subset_generator)[
            : self.target_subset_size
        ]
        return per_head[subset.to(per_head.device)].min(dim=0).values

    def update(
        self,
        replay: TransitionReplayBuffer,
        *,
        target_policy: TanhPolicyProtocol,
        entropy_rate: float,
    ) -> dict[str, float]:
        """Apply one averaged optimizer step over disjoint per-head batches.

        Heads learn from disjoint slices of a single draw rather than from one
        replay sample each: the data stays independent across heads and the
        update costs one sample and one next-action pass instead of ``M``.
        """
        batch = replay.sample(
            self.num_heads * self.train_cfg.critic_batch_size,
            generator=self.replay_generator,
            device=self.device,
        )
        losses, targets = self._per_head_losses(
            batch, target_policy=target_policy, entropy_rate=entropy_rate
        )
        loss = torch.stack(losses).mean()
        online_gradient_norm = _optimize(
            loss,
            parameters=self.online.parameters(),
            optimizer=self.optimizer,
            max_grad_norm=self.train_cfg.max_grad_norm,
            name="direct-Q TD loss",
        )
        polyak_update(self.online, self.target, self.train_cfg.target_polyak)
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

    def _per_head_losses(
        self,
        batch: TransitionBatch,
        *,
        target_policy: TanhPolicyProtocol,
        entropy_rate: float,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """TD loss and mean target of each head, on its own slice of the draw."""
        batch_size = self.train_cfg.critic_batch_size
        with torch.no_grad():
            next_actions, next_log_probability = (
                target_policy.sample_with_log_probability(
                    batch.next_policy_observations, generator=self.action_generator
                )
            )
            rewards = self.reward.score(batch.features)
            # Every target head over the whole draw, so each head's slice can
            # bootstrap from a clipped minimum rather than from itself alone.
            next_q_per_head = self.target(batch.next_critic_observations, next_actions)

        losses: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for head_index in range(self.num_heads):
            rows = head_rows(head_index, batch_size)
            with torch.no_grad():
                target = soft_task_q_td_target(
                    rewards[rows].to(next_q_per_head),
                    next_log_probability[rows],
                    self._bootstrap_value(next_q_per_head[:, rows], head_index),
                    batch.bootstrap_valid[rows],
                    gamma=self.train_cfg.gamma,
                    entropy_rate=entropy_rate,
                    step_dt=self.step_dt,
                )
            prediction = self.online.head(
                head_index, batch.critic_observations[rows], batch.actions[rows]
            )
            losses.append(F.huber_loss(prediction, target))
            targets.append(target.mean())
        return losses, targets

    def mean(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        entropy_rate: float,
    ) -> ValueEstimate:
        """Score actions [B,A] for control, reducing the ensemble as configured."""
        del entropy_rate
        per_head = self.online(observations, actions)
        q = _reduce_heads(per_head, self.control_reduction)
        zeros = torch.zeros_like(q)
        return ValueEstimate(q=q, reward_q=q, entropy_q=zeros)

    def modules(self) -> tuple[nn.Module, ...]:
        return self.online, self.target

    def reset_parameters(self) -> None:
        """Reinitialize the critics and their optimizer, keeping replay."""
        self.online.reset_parameters()
        self.target = self.online.target_copy()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.train_cfg.critic_learning_rate
        )

    def save_state(self) -> dict[str, Any]:
        return {
            "type": "direct",
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_rng": self.replay_generator.get_state(),
            "action_rng": self.action_generator.get_state(),
            "subset_rng": self.subset_generator.get_state(),
        }

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool) -> None:
        if state.get("type") != "direct":
            raise ValueError("Checkpoint value backend is not direct Q.")
        self.online.load_state_dict(state["online"], strict=True)
        self.target.load_state_dict(state["target"], strict=True)
        self.replay_generator.set_state(state["replay_rng"].cpu())
        self.action_generator.set_state(state["action_rng"].cpu())
        self.subset_generator.set_state(state["subset_rng"].cpu())
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
        reward: LinearRewardScorer,
        cfg: SuccessorQCfg,
        train_cfg: OffPolicyTrainCfg,
        use_entropy_critic: bool,
        step_dt: float,
        device: torch.device,
        seed: int,
        horizon: int | None = None,
    ) -> None:
        if reward.feature_dim != feature_dim:
            raise ValueError("Reward and successor feature dimensions must match.")
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be > 0, got {step_dt}.")
        self.num_heads = cfg.num_heads
        self.control_quantile = cfg.control_quantile
        self.reward = reward
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.device = device
        # Only needed when nothing discounts: then the episode is what bounds
        # the successor sum, and its scale.
        self.horizon = horizon
        self.online = SuccessorFeatureEnsemble(
            observation_dim,
            action_dim,
            feature_dim,
            num_heads=cfg.num_heads,
            width=cfg.width,
            residual_blocks=cfg.residual_blocks,
            latent_dim=cfg.latent_dim,
            head_hidden_dim=cfg.head_hidden_dim,
        ).to(device)
        self.target = self.online.target_copy()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=train_cfg.critic_learning_rate
        )
        self.entropy: EntropyReturnCritic | None = (
            EntropyReturnCritic(
                observation_dim=observation_dim,
                action_dim=action_dim,
                cfg=cfg,
                train_cfg=train_cfg,
                step_dt=step_dt,
                device=device,
                seed=seed,
            )
            if use_entropy_critic
            else None
        )
        self.return_scale = torch.ones(feature_dim, device=device)
        self.scale_fitted = False
        self.scale_refits = 0
        self._feature_square_sum = torch.zeros(feature_dim, device=device)
        self._feature_count = 0
        self.replay_generator = torch.Generator(device="cpu").manual_seed(seed + 1009)
        self.action_generator = torch.Generator(device=device).manual_seed(seed + 2003)

    @torch.no_grad()
    def fit_feature_scale(self, rollout: TransitionRollout) -> None:
        """Keep the successor output scale matched to the observed features.

        Features a competent policy produces (a lifted object, a reached goal)
        are identically zero in the first rollouts, so a scale fitted once there
        stays at its floor and those heads have to emit values orders of
        magnitude off their unit range. The running second moment tracks them,
        and a refit rescales the heads so predictions are unchanged.
        """
        features = rearrange(rollout.features, "n t d -> (n t) d").to(self.device)
        self._feature_square_sum += features.square().sum(dim=0)
        self._feature_count += features.shape[0]
        observed_scale = successor_return_scale_from_second_moment(
            self._feature_square_sum / self._feature_count,
            gamma=self.train_cfg.gamma,
            minimum_scale=self.cfg.return_scale_min,
            horizon=self.horizon,
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
        entropy_rate: float,
    ) -> dict[str, float]:
        if not self.scale_fitted:
            raise RuntimeError("Fit successor feature scales before TD updates.")
        batch = replay.sample(
            self.num_heads * self.train_cfg.critic_batch_size,
            generator=self.replay_generator,
            device=self.device,
        )
        losses, feature_losses = self._per_head_losses(
            batch, target_policy=target_policy
        )
        loss = torch.stack(losses).mean()
        online_gradient_norm = _optimize(
            loss,
            parameters=self.online.parameters(),
            optimizer=self.optimizer,
            max_grad_norm=self.train_cfg.max_grad_norm,
            name="successor TD loss",
        )
        polyak_update(self.online, self.target, self.train_cfg.target_polyak)
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
        if self.entropy is not None:
            metrics.update(self.entropy.update(replay, target_policy=target_policy))
        elif entropy_rate != 0.0:
            raise RuntimeError(
                "A positive entropy rate needs an entropy-return critic; "
                "this backend was built without one."
            )
        return metrics

    def _per_head_losses(
        self, batch: TransitionBatch, *, target_policy: TanhPolicyProtocol
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Scale-normalized TD loss of each head, and its per-feature breakdown.

        The targets contain features and dynamics only - no reward weights -
        which is what lets a changing learned reward leave them untouched.
        """
        batch_size = self.train_cfg.critic_batch_size
        with torch.no_grad():
            next_actions, _ = target_policy.sample_with_log_probability(
                batch.next_policy_observations, generator=self.action_generator
            )
            # One trunk pass for the whole batch; each head reads its own slice.
            next_latent = self.target.latent(
                batch.next_critic_observations, next_actions
            )
        current_latent = self.online.latent(batch.critic_observations, batch.actions)

        losses: list[torch.Tensor] = []
        feature_losses: list[torch.Tensor] = []
        for head_index in range(self.num_heads):
            rows = head_rows(head_index, batch_size)
            with torch.no_grad():
                target = successor_ensemble_td_target(
                    batch.features[rows],
                    self.target.apply_head(head_index, next_latent[rows]),
                    batch.bootstrap_valid[rows],
                    self.train_cfg.gamma,
                )
            prediction = self.online.apply_head(head_index, current_latent[rows])
            residual = (prediction - target) / rearrange(self.return_scale, "d -> 1 d")
            by_feature = F.huber_loss(
                residual, torch.zeros_like(residual), reduction="none"
            ).mean(dim=0)
            losses.append(by_feature.mean())
            feature_losses.append(by_feature.detach())
        return losses, feature_losses

    def mean(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        entropy_rate: float,
    ) -> ValueEstimate:
        # [M,B,D] successor features -> one scalar per head -> reduction.
        successors = self.online(observations, actions)
        reward_weights = self.reward.weights.detach().to(successors)
        per_head_reward_q = self.reward.step_dt.to(successors) * einsum(
            successors, reward_weights, "m b d, d -> m b"
        )
        reward_q = _ensemble_quantile(per_head_reward_q, self.control_quantile)
        entropy_q = (
            torch.zeros_like(reward_q)
            if self.entropy is None
            else self.entropy.value(observations, actions)
        )
        combined_q = reward_q + float(entropy_rate) * entropy_q
        _require_finite(combined_q, name="successor scalarized Q")
        return ValueEstimate(
            q=combined_q,
            reward_q=reward_q,
            entropy_q=entropy_q,
        )

    def reset_parameters(self) -> None:
        """Reinitialize successor and entropy critics, keeping the feature scale."""
        self.online.reset_parameters()
        self.online.set_output_scale(self.return_scale)
        self.target = self.online.target_copy()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(), lr=self.train_cfg.critic_learning_rate
        )
        if self.entropy is not None:
            self.entropy.reset_parameters()

    def modules(self) -> tuple[nn.Module, ...]:
        modules: list[nn.Module] = [self.online, self.target]
        if self.entropy is not None:
            modules.extend(self.entropy.modules())
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
            "replay_rng": self.replay_generator.get_state(),
            "action_rng": self.action_generator.get_state(),
        }
        if self.entropy is not None:
            state["entropy"] = self.entropy.save_state()
        return state

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool) -> None:
        if state.get("type") != "successor":
            raise ValueError("Checkpoint value backend is not successor Q.")
        checkpoint_has_entropy = "entropy" in state
        if checkpoint_has_entropy != (self.entropy is not None):
            raise ValueError(
                "Checkpoint entropy-critic configuration differs: checkpoint "
                f"{'has' if checkpoint_has_entropy else 'has no'} entropy critic, "
                f"this backend {'has' if self.entropy is not None else 'has none'}."
            )
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
        self.replay_generator.set_state(state["replay_rng"].cpu())
        self.action_generator.set_state(state["action_rng"].cpu())
        if self.entropy is not None:
            self.entropy.load_state(state["entropy"], load_optimizer=load_optimizer)
        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])


def _reduce_heads(per_head: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduce per-head values ``[M,B]`` to the value control acts on ``[B]``."""
    if reduction == "mean":
        return per_head.mean(dim=0)
    if reduction == "min":
        return per_head.min(dim=0).values
    raise ValueError(f"Unknown control reduction {reduction!r}.")


def _ensemble_quantile(per_head: torch.Tensor, quantile: float) -> torch.Tensor:
    """Reduce per-head values ``[M,B]`` to the score control acts on ``[B]``.

    Below 0.5 the score is pessimistic, above it optimistic: the actor is drawn
    toward actions whose value the ensemble disagrees about.
    """
    if per_head.ndim != 2:
        raise ValueError(
            f"Expected per-head values [M,B], got {tuple(per_head.shape)}."
        )
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0.0, 1.0], got {quantile}.")
    return torch.quantile(per_head, float(quantile), dim=0, interpolation="linear")


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
