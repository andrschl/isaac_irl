"""Exponential policy mirror descent with direct or successor action values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange

from algorithms.bro_network import reset_module_parameters
from algorithms.exponential_policy import (
    ExponentialTarget,
    adaptive_exponential_target,
)
from algorithms.off_policy_common import (
    BehaviorCloningLoss,
    copy_parameters,
    decomposed_policy_kl,
    gradient_norm,
    interpolate_parameters,
    mix_actor_loss,
    parameter_delta,
    repeat_observations,
)
from algorithms.state_dependent_tanh_actor import TanhPolicyProtocol
from algorithms.value_backends import ActionValueBackend, ValueEstimate
from config.off_policy import PmdCfg
from storage.transition_replay import (
    Observation,
    TransitionReplayBuffer,
    TransitionRollout,
)


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    """One frozen exponential target over K candidates per replay state.

    Candidates are kept as pre-squash latents: the projection scores them under
    the fitted policy, and inverting tanh would lose the ones near the bound.

    Shapes:
        latent_actions: [B, K, A]
        reference_log_probability: [B, K]
        value / target tensors: [B, K]
    """

    latent_actions: torch.Tensor
    reference_log_probability: torch.Tensor
    value: ValueEstimate
    target: ExponentialTarget


class Pmd:
    """Fit a neural actor to pi_old**beta * exp(q_coefficient * Q)."""

    def __init__(
        self,
        policy: TanhPolicyProtocol,
        value: ActionValueBackend,
        *,
        cfg: PmdCfg,
        device: torch.device,
        bc: BehaviorCloningLoss | None = None,
        seed: int = 0,
    ) -> None:
        self.policy = policy
        self.previous_policy = policy.snapshot()
        self.value = value
        self.cfg = cfg
        self.device = device
        self.bc = bc
        self.actor_optimizer = torch.optim.Adam(
            policy.parameters(), lr=cfg.actor_learning_rate
        )
        self.replay_generator = torch.Generator(device="cpu").manual_seed(seed + 4001)
        self.action_generator = torch.Generator(device=device).manual_seed(seed + 4007)
        self.policy_version = 0

    @property
    def entropy_temperature(self) -> float:
        return float(self.cfg.entropy_temperature)

    def observe_rollout(self, rollout: TransitionRollout) -> None:
        """Fit/update value-side statistics from one newly collected rollout."""
        self.value.fit_feature_scale(rollout)

    def reset_parameters(self) -> None:
        """Reinitialize the critics, and the actor unless the scope excludes it.

        The frozen policy is re-taken from whatever the actor is afterwards:
        projecting away from a policy that no longer exists would measure a KL
        against a deleted network.
        """
        self.value.reset_parameters()
        if self.cfg.train.reset_scope == "critic":
            self.previous_policy = self.policy.snapshot()
            return
        reset_module_parameters(self.policy.model)
        self.previous_policy = self.policy.snapshot()
        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.cfg.actor_learning_rate
        )

    def critic_update(self, replay: TransitionReplayBuffer) -> dict[str, float]:
        """Evaluate the frozen pi_k that the next projection improves upon."""
        return self.value.update(
            replay,
            target_policy=self.previous_policy,
            entropy_temperature=self.entropy_temperature,
        )

    @torch.no_grad()
    def _candidate_batch(
        self,
        policy_observations: Observation,
        critic_observations: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> CandidateBatch:
        """Draw K actions per state from pi_old and score them once."""
        batch_size = critic_observations.shape[0]
        candidate_count = self.cfg.candidate_actions
        repeated_policy_observations = repeat_observations(
            policy_observations, candidate_count
        )
        actions, latent_actions, reference_log_probability = (
            self.previous_policy.sample_with_latent(
                repeated_policy_observations,
                generator=generator,
            )
        )
        flat_critic_observations = critic_observations.repeat_interleave(
            candidate_count, dim=0
        )
        flat_value = self.value.mean(
            flat_critic_observations,
            actions,
            entropy_temperature=self.entropy_temperature,
        )

        # Candidates are stored consecutively per state: [(B K), ...] -> [B, K, ...].
        def unflatten(values: torch.Tensor) -> torch.Tensor:
            return rearrange(values, "(b k) -> b k", b=batch_size, k=candidate_count)

        reshaped_log_probability = unflatten(reference_log_probability)
        scalar_q = unflatten(flat_value.q)
        target = adaptive_exponential_target(
            scalar_q,
            reshaped_log_probability,
            beta=self.cfg.beta,
            target_kl=self.cfg.candidate_target_kl,
            coefficient_min=self.cfg.q_coefficient_min,
            coefficient_max=self.cfg.q_coefficient_max,
            bisection_steps=self.cfg.coefficient_bisection_steps,
        )
        return CandidateBatch(
            latent_actions=rearrange(
                latent_actions, "(b k) a -> b k a", b=batch_size, k=candidate_count
            ),
            reference_log_probability=reshaped_log_probability,
            value=ValueEstimate(
                q=scalar_q,
                reward_q=unflatten(flat_value.reward_q),
                entropy_q=unflatten(flat_value.entropy_q),
            ),
            target=target,
        )

    def _projection_loss(
        self,
        repeated_observations: Observation,
        candidates: CandidateBatch,
    ) -> torch.Tensor:
        """Weighted negative log likelihood of the candidate target.

        ``repeated_observations`` holds each replay state repeated K times
        consecutively, matching the candidate layout [B, K, A].
        """
        batch_size, candidate_count, _ = candidates.latent_actions.shape
        flat_log_probability = self.policy.log_probability_from_latent(
            repeated_observations,
            rearrange(candidates.latent_actions, "b k a -> (b k) a"),
        )
        log_probability = rearrange(
            flat_log_probability, "(b k) -> b k", b=batch_size, k=candidate_count
        )
        return -(candidates.target.weights.detach() * log_probability).mean()

    def _projection_kl(self, policy_observations: Observation) -> tuple[float, float]:
        """Mean and spread KL of the fitted actor from the frozen policy."""
        with torch.no_grad():
            mean_kl, spread_kl = decomposed_policy_kl(
                self.previous_policy, self.policy, policy_observations
            )
        if not torch.isfinite(mean_kl) or not torch.isfinite(spread_kl):
            raise FloatingPointError("PMD projection KL is non-finite.")
        return float(mean_kl.item()), float(spread_kl.item())

    def _leaves_trust_region(self, mean_kl: float, spread_kl: float) -> bool:
        """Whether either part of the step exceeds its own bound."""
        return (
            mean_kl + spread_kl > self.cfg.actor_projection_kl
            or spread_kl > self.cfg.actor_spread_projection_kl
        )

    def _backtrack_into_trust_region(
        self,
        start_parameters: dict[str, torch.Tensor],
        policy_observations: Observation,
    ) -> tuple[float, float, int]:
        """Halve the accumulated displacement until the KL bound holds.

        One optimizer step can overshoot the bound by orders of magnitude, so
        the bound is a guarantee only if the step that crossed it is shrunk.
        The Adam moments of the discarded step are kept: the next projection
        measures its own KL, so stale moments cost at most one more backtrack.
        Returns the realized KL and how many halvings were needed.
        """
        proposed = copy_parameters(self.policy.model)
        factor = 1.0
        for backtracks in range(1, self.cfg.projection_backtrack_steps + 1):
            factor *= 0.5
            interpolate_parameters(
                self.policy.model, start_parameters, proposed, factor
            )
            mean_kl, spread_kl = self._projection_kl(policy_observations)
            if not self._leaves_trust_region(mean_kl, spread_kl):
                return mean_kl, spread_kl, backtracks
        # Nothing on the ray satisfies the bounds: keep the frozen policy.
        interpolate_parameters(self.policy.model, start_parameters, proposed, 0.0)
        mean_kl, spread_kl = self._projection_kl(policy_observations)
        return mean_kl, spread_kl, self.cfg.projection_backtrack_steps

    def actor_update(self, replay: TransitionReplayBuffer) -> dict[str, float]:
        """Project the actor onto one frozen target, stopping at the KL bound."""
        batch = replay.sample(
            self.cfg.actor_batch_size,
            generator=self.replay_generator,
            device=self.device,
        )
        before = copy_parameters(self.policy.model)
        projection_kl = 0.0
        spread_kl = 0.0
        projection_steps_taken = 0
        projection_backtracks = 0
        last_gradient_norm = 0.0
        bc_loss_value = 0.0

        candidates = self._candidate_batch(
            batch.policy_observations,
            batch.critic_observations,
            generator=self.action_generator,
        )
        repeated_observations = repeat_observations(
            batch.policy_observations, self.cfg.candidate_actions
        )
        with torch.no_grad():
            initial_loss = float(
                self._projection_loss(repeated_observations, candidates).item()
            )
        for _ in range(self.cfg.projection_steps):
            rl_loss = self._projection_loss(repeated_observations, candidates)
            actor_loss, bc_loss = mix_actor_loss(
                rl_loss, policy=self.policy, bc=self.bc
            )
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("PMD actor loss is non-finite.")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            last_gradient_norm = gradient_norm(self.policy.parameters())
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.max_grad_norm
            )
            self.actor_optimizer.step()
            projection_steps_taken += 1
            bc_loss_value = float(bc_loss.item())
            projection_kl, spread_kl = self._projection_kl(batch.policy_observations)
            if self._leaves_trust_region(projection_kl, spread_kl):
                projection_kl, spread_kl, projection_backtracks = (
                    self._backtrack_into_trust_region(before, batch.policy_observations)
                )
                break
        with torch.no_grad():
            final_loss = float(
                self._projection_loss(repeated_observations, candidates).item()
            )
            _, policy_std = self.policy.distribution_parameters(
                batch.policy_observations
            )

        self.previous_policy = self.policy.snapshot()
        self.policy_version += 1
        probabilities = candidates.target.probabilities
        candidate_entropy = (
            -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=1).mean()
        )
        bc_weight = 0.0 if self.bc is None else float(self.bc.weight)
        mixed_loss = (1.0 - bc_weight) * final_loss + bc_weight * bc_loss_value
        return {
            "Actor/loss": mixed_loss,
            "Actor/rl_loss": final_loss,
            "Actor/bc_loss": bc_loss_value,
            "Actor/bc_weight": bc_weight,
            "Actor/gradient_norm": last_gradient_norm,
            "Actor/parameter_delta": parameter_delta(before, self.policy.model),
            "Actor/policy_std_mean": float(policy_std.mean().item()),
            "PMD/beta": float(self.cfg.beta),
            "PMD/q_coefficient": candidates.target.q_coefficient,
            "PMD/candidate_target_kl": candidates.target.candidate_kl,
            "PMD/actor_projection_kl": projection_kl + spread_kl,
            "PMD/actor_mean_kl": projection_kl,
            "PMD/actor_spread_kl": spread_kl,
            "PMD/effective_sample_size": candidates.target.effective_sample_size,
            "PMD/candidate_entropy": float(candidate_entropy.item()),
            "PMD/projection_loss_before": initial_loss,
            "PMD/projection_loss_after": final_loss,
            "PMD/projection_steps_taken": float(projection_steps_taken),
            "PMD/projection_backtracks": float(projection_backtracks),
            "Value/q_mean": float(candidates.value.q.mean().item()),
            "Value/reward_q_mean": float(candidates.value.reward_q.mean().item()),
            "Value/entropy_q_mean": float(candidates.value.entropy_q.mean().item()),
        }

    def train(self) -> None:
        self.policy.train()
        for module in self.value.modules():
            module.train()

    def eval(self) -> None:
        self.policy.eval()
        for module in self.value.modules():
            module.eval()

    def save_state(self) -> dict[str, Any]:
        return {
            "algorithm": "pmd",
            "policy": self.policy.model.state_dict(),
            "previous_policy": self.previous_policy.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value": self.value.save_state(),
            "replay_rng": self.replay_generator.get_state(),
            "action_rng": self.action_generator.get_state(),
            "policy_version": self.policy_version,
        }

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool = True) -> None:
        if state.get("algorithm") != "pmd":
            raise ValueError("Checkpoint does not contain the PMD learner.")
        self.policy.model.load_state_dict(state["policy"], strict=True)
        self.previous_policy = self.policy.snapshot()
        self.previous_policy.model.load_state_dict(
            state["previous_policy"], strict=True
        )
        self.value.load_state(state["value"], load_optimizer=load_optimizer)
        self.replay_generator.set_state(state["replay_rng"].cpu())
        self.action_generator.set_state(state["action_rng"].cpu())
        self.policy_version = int(state["policy_version"])
        if load_optimizer:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
