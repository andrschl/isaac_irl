"""Reparameterized SAC with a selectable direct or successor value backend."""

from __future__ import annotations

from typing import Any

import torch

from algorithms.off_policy_common import (
    BehaviorCloningLoss,
    copy_parameters,
    frozen_modules,
    gradient_norm,
    mix_actor_loss,
    parameter_delta,
    update_observation_normalizers,
)
from algorithms.state_dependent_tanh_actor import TanhPolicyProtocol
from algorithms.successor_ensemble import RunningStandardizer
from algorithms.value_backends import ActionValueBackend
from config.off_policy import SacCfg
from storage.transition_replay import TransitionReplayBuffer, TransitionRollout


class Sac:
    """SAC policy improvement; deliberately owns no previous-policy snapshot."""

    def __init__(
        self,
        policy: TanhPolicyProtocol,
        value: ActionValueBackend,
        *,
        actor_observation_normalizer: RunningStandardizer,
        critic_observation_normalizer: RunningStandardizer,
        cfg: SacCfg,
        action_dim: int,
        device: torch.device,
        bc: BehaviorCloningLoss | None = None,
        seed: int = 0,
    ) -> None:
        self.policy = policy
        self.value = value
        self.actor_observation_normalizer = actor_observation_normalizer
        self.critic_observation_normalizer = critic_observation_normalizer
        self.cfg = cfg
        self.device = device
        self.bc = bc
        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.actor_learning_rate
        )
        self.actor_replay_generator = torch.Generator(device="cpu").manual_seed(
            seed + 3001
        )
        self.actor_action_generator = torch.Generator(device=device).manual_seed(
            seed + 3007
        )
        initial_temperature = max(cfg.initial_entropy_temperature, 1.0e-12)
        self.log_entropy_temperature = (
            torch.nn.Parameter(
                torch.tensor(
                    initial_temperature,
                    device=device,
                    dtype=torch.float32,
                ).log()
            )
            if cfg.automatic_entropy_tuning
            else None
        )
        self.entropy_optimizer = (
            torch.optim.Adam(
                [self.log_entropy_temperature],
                lr=cfg.entropy_learning_rate,
            )
            if self.log_entropy_temperature is not None
            else None
        )
        self.target_entropy = -float(action_dim)
        self.policy_version = 0

    @property
    def entropy_temperature(self) -> float:
        if self.log_entropy_temperature is None:
            return float(self.cfg.initial_entropy_temperature)
        return float(self.log_entropy_temperature.detach().exp().item())

    def observe_rollout(self, rollout: TransitionRollout) -> None:
        """Fit/update statistics from one newly collected rollout."""
        update_observation_normalizers(
            rollout,
            actor=self.actor_observation_normalizer,
            critic=self.critic_observation_normalizer,
        )
        self.value.fit_feature_scale(rollout)

    def critic_update(self, replay: TransitionReplayBuffer) -> dict[str, float]:
        return self.value.update(
            replay,
            target_policy=self.policy,
            entropy_temperature=self.entropy_temperature,
        )

    def actor_update(self, replay: TransitionReplayBuffer) -> dict[str, float]:
        """Apply one reparameterized SAC update on replay states [B,O]."""
        batch = replay.sample(
            self.cfg.actor_batch_size,
            generator=self.actor_replay_generator,
            device=self.device,
        )
        before = copy_parameters(self.policy.model)
        with frozen_modules(self.value.modules()):
            actions, log_probability = self.policy.rsample_with_log_probability(
                batch.policy_observations,
                generator=self.actor_action_generator,
            )
            estimate = self.value.mean(
                batch.critic_observations,
                actions,
                entropy_temperature=self.entropy_temperature,
            )
            rl_loss = (self.entropy_temperature * log_probability - estimate.q).mean()
            actor_loss, bc_loss = mix_actor_loss(
                rl_loss, policy=self.policy, bc=self.bc
            )
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("SAC actor loss is non-finite.")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient_norm = gradient_norm(self.policy.parameters())
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.max_grad_norm
            )
            self.actor_optimizer.step()

        entropy_loss = self._update_entropy_temperature(log_probability)
        with torch.no_grad():
            _, policy_std = self.policy.distribution_parameters(
                batch.policy_observations
            )
        self.policy_version += 1
        bc_weight = 0.0 if self.bc is None else float(self.bc.weight)
        return {
            "Actor/loss": float(actor_loss.detach().item()),
            "Actor/rl_loss": float(rl_loss.detach().item()),
            "Actor/bc_loss": float(bc_loss.item()),
            "Actor/bc_weight": bc_weight,
            "Actor/gradient_norm": actor_gradient_norm,
            "Actor/parameter_delta": parameter_delta(before, self.policy.model),
            "Actor/policy_std_mean": float(policy_std.mean().item()),
            "SAC/entropy_temperature": self.entropy_temperature,
            "SAC/entropy_temperature_loss": entropy_loss,
            "Value/q_mean": float(estimate.q.detach().mean().item()),
            "Value/reward_q_mean": float(estimate.reward_q.detach().mean().item()),
            "Value/entropy_q_mean": float(estimate.entropy_q.detach().mean().item()),
        }

    def _update_entropy_temperature(self, log_probability: torch.Tensor) -> float:
        if self.log_entropy_temperature is None or self.entropy_optimizer is None:
            return 0.0
        loss = -(
            self.log_entropy_temperature
            * (log_probability.detach() + self.target_entropy)
        ).mean()
        self.entropy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.entropy_optimizer.step()
        return float(loss.detach().item())

    def train(self) -> None:
        self.policy.train()
        for module in self.value.modules():
            module.train()

    def eval(self) -> None:
        self.policy.eval()
        for module in self.value.modules():
            module.eval()

    def save_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "algorithm": "sac",
            "policy": self.policy.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value": self.value.save_state(),
            "actor_normalizer": self.actor_observation_normalizer.state_dict(),
            "critic_normalizer": self.critic_observation_normalizer.state_dict(),
            "actor_replay_rng": self.actor_replay_generator.get_state(),
            "actor_action_rng": self.actor_action_generator.get_state(),
            "policy_version": self.policy_version,
        }
        if self.log_entropy_temperature is not None:
            if self.entropy_optimizer is None:
                raise RuntimeError("Missing entropy-temperature optimizer.")
            state["log_entropy_temperature"] = (
                self.log_entropy_temperature.detach().cpu()
            )
            state["entropy_optimizer"] = self.entropy_optimizer.state_dict()
        return state

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool = True) -> None:
        if state.get("algorithm") != "sac":
            raise ValueError("Checkpoint does not contain the SAC learner.")
        self.policy.model.load_state_dict(state["policy"], strict=True)
        self.value.load_state(state["value"], load_optimizer=load_optimizer)
        self.actor_observation_normalizer.load_state_dict(
            state["actor_normalizer"], strict=True
        )
        self.critic_observation_normalizer.load_state_dict(
            state["critic_normalizer"], strict=True
        )
        self.actor_replay_generator.set_state(state["actor_replay_rng"].cpu())
        self.actor_action_generator.set_state(state["actor_action_rng"].cpu())
        self.policy_version = int(state["policy_version"])
        if self.log_entropy_temperature is not None:
            if "log_entropy_temperature" not in state:
                raise ValueError("Checkpoint is missing entropy-temperature state.")
            self.log_entropy_temperature.data.copy_(
                state["log_entropy_temperature"].to(self.device)
            )
            if load_optimizer:
                if self.entropy_optimizer is None:
                    raise RuntimeError("Missing entropy-temperature optimizer.")
                self.entropy_optimizer.load_state_dict(state["entropy_optimizer"])
        if load_optimizer:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
