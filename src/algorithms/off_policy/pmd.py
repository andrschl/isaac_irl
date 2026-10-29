"""Exponential policy mirror descent with direct or successor action values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange

from algorithms.off_policy.actor import TanhPolicyProtocol
from algorithms.off_policy.common import (
    BehaviorCloningLoss,
    copy_parameters,
    decomposed_policy_kl,
    gradient_norm,
    interpolate_parameters,
    mix_actor_loss,
    parameter_delta,
    repeat_observations,
)
from algorithms.off_policy.exponential_target import (
    ExponentialTarget,
    adaptive_exponential_target,
)
from algorithms.off_policy.values.backends import ActionValueBackend, ValueEstimate
from algorithms.off_policy.values.networks import reset_module_parameters
from config.schema import PmdCfg
from storage.observations import Observation, map_observation
from storage.transition_replay import (
    TransitionReplayBuffer,
    TransitionRollout,
)


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """What one bounded projection of the actor achieved.

    ``mean_kl`` and ``spread_kl`` split the step into "moved the actions" and
    "changed how noisy they are", because one bound on the whole Gaussian
    cannot tell those apart and a weighted fit to its own samples drifts
    outward.
    """

    mean_kl: float
    spread_kl: float
    steps_taken: int
    backtracks: int
    gradient_norm: float
    bc_loss: float


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
        step_dt: float,
        device: torch.device,
        bc: BehaviorCloningLoss | None = None,
        seed: int = 0,
    ) -> None:
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be > 0, got {step_dt}.")
        self.policy = policy
        self.previous_policy = policy.snapshot()
        self.value = value
        self.cfg = cfg
        self.step_dt = float(step_dt)
        self.device = device
        self.bc = bc
        self.actor_optimizer = torch.optim.Adam(
            policy.parameters(), lr=cfg.actor_learning_rate
        )
        self.replay_generator = torch.Generator(device="cpu").manual_seed(seed + 4001)
        self.action_generator = torch.Generator(device=device).manual_seed(seed + 4007)
        self.policy_version = 0
        self._collection_policy: TanhPolicyProtocol | None = None
        self._collection_observations: Observation | None = None

    @property
    def entropy_rate(self) -> float:
        return float(self.cfg.entropy_rate)

    @property
    def entropy_step_coefficient(self) -> float:
        """Coefficient on one step's ``-log pi`` in the common soft MDP."""
        return self.step_dt * self.entropy_rate

    def observe_rollout(self, rollout: TransitionRollout) -> None:
        """Fit/update value-side statistics from one newly collected rollout."""
        self.value.fit_feature_scale(rollout)
        self._collection_policy = self.policy.snapshot()
        # Evenly spaced retained states [B,O], without consuming training RNGs.
        sample_count = min(rollout.transition_count, self.cfg.actor_batch_size)

        def sample_states(leaf: torch.Tensor) -> torch.Tensor:
            states = rearrange(leaf, "n t ... -> (n t) ...")
            indices = torch.linspace(
                0, states.shape[0] - 1, sample_count, device=states.device
            ).long()
            return states[indices].detach().to(self.device)

        self._collection_observations = map_observation(
            rollout.policy_observations, sample_states
        )

    @torch.no_grad()
    def _collection_policy_metrics(self) -> dict[str, float]:
        """Cumulative KL(collection policy || current) on fixed collected states."""
        if self._collection_policy is None or self._collection_observations is None:
            return {}
        mean_kl, spread_kl = decomposed_policy_kl(
            self._collection_policy, self.policy, self._collection_observations
        )
        return {
            "PMD/collection_policy_kl": float(mean_kl + spread_kl),
            "PMD/collection_policy_mean_kl": float(mean_kl),
            "PMD/collection_policy_spread_kl": float(spread_kl),
        }

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
            entropy_rate=self.entropy_rate,
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
            entropy_rate=self.entropy_rate,
        )

        # Candidates are stored consecutively per state: [(B K), ...] -> [B, K, ...].
        def unflatten(values: torch.Tensor) -> torch.Tensor:
            return rearrange(values, "(b k) -> b k", b=batch_size, k=candidate_count)

        reshaped_log_probability = unflatten(reference_log_probability)
        scalar_q = unflatten(flat_value.q)
        # Q(s,a) contains entropy from the next policy action onward. The
        # current action's dt-scaled entropy belongs in policy improvement.
        improvement_q = (
            scalar_q - self.entropy_step_coefficient * reshaped_log_probability
        )
        target = adaptive_exponential_target(
            improvement_q,
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
        """Project the actor onto one frozen candidate target and report it."""
        batch = replay.sample(
            self.cfg.actor_batch_size,
            generator=self.replay_generator,
            device=self.device,
        )
        before = copy_parameters(self.policy.model)
        candidates = self._candidate_batch(
            batch.policy_observations,
            batch.critic_observations,
            generator=self.action_generator,
        )
        repeated_observations = repeat_observations(
            batch.policy_observations, self.cfg.candidate_actions
        )
        with torch.no_grad():
            loss_before = float(
                self._projection_loss(repeated_observations, candidates).item()
            )

        result = self._project(
            batch.policy_observations, repeated_observations, candidates, before
        )

        with torch.no_grad():
            loss_after = float(
                self._projection_loss(repeated_observations, candidates).item()
            )
            _, policy_std = self.policy.distribution_parameters(
                batch.policy_observations
            )

        self.previous_policy = self.policy.snapshot()
        self.policy_version += 1
        return self._metrics(
            candidates=candidates,
            result=result,
            before=before,
            policy_std=policy_std,
            loss_before=loss_before,
            loss_after=loss_after,
        )

    def _project(
        self,
        policy_observations: Observation,
        repeated_observations: Observation,
        candidates: CandidateBatch,
        start_parameters: dict[str, torch.Tensor],
    ) -> ProjectionResult:
        """Fit the actor to the candidate target, stopping at the KL bound."""
        mean_kl = spread_kl = 0.0
        gradient_norm_value = 0.0
        bc_loss_value = 0.0
        backtracks = 0

        for step in range(1, self.cfg.projection_steps + 1):
            rl_loss = self._projection_loss(repeated_observations, candidates)
            actor_loss, bc_loss = mix_actor_loss(
                rl_loss, policy=self.policy, bc=self.bc
            )
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("PMD actor loss is non-finite.")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            gradient_norm_value = gradient_norm(self.policy.parameters())
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.max_grad_norm
            )
            self.actor_optimizer.step()
            bc_loss_value = float(bc_loss.item())

            mean_kl, spread_kl = self._projection_kl(policy_observations)
            if self._leaves_trust_region(mean_kl, spread_kl):
                mean_kl, spread_kl, backtracks = self._backtrack_into_trust_region(
                    start_parameters, policy_observations
                )
                return ProjectionResult(
                    mean_kl=mean_kl,
                    spread_kl=spread_kl,
                    steps_taken=step,
                    backtracks=backtracks,
                    gradient_norm=gradient_norm_value,
                    bc_loss=bc_loss_value,
                )
        return ProjectionResult(
            mean_kl=mean_kl,
            spread_kl=spread_kl,
            steps_taken=self.cfg.projection_steps,
            backtracks=backtracks,
            gradient_norm=gradient_norm_value,
            bc_loss=bc_loss_value,
        )

    def _metrics(
        self,
        *,
        candidates: CandidateBatch,
        result: ProjectionResult,
        before: dict[str, torch.Tensor],
        policy_std: torch.Tensor,
        loss_before: float,
        loss_after: float,
    ) -> dict[str, float]:
        """Report one projection: the actor, the target it aimed at, and Q."""
        bc_weight = 0.0 if self.bc is None else float(self.bc.weight)
        target = candidates.target
        return {
            "Actor/loss": (1.0 - bc_weight) * loss_after + bc_weight * result.bc_loss,
            "Actor/rl_loss": loss_after,
            "Actor/bc_loss": result.bc_loss,
            "Actor/bc_weight": bc_weight,
            "Actor/gradient_norm": result.gradient_norm,
            "Actor/parameter_delta": parameter_delta(before, self.policy.model),
            "Actor/policy_std_mean": float(policy_std.mean().item()),
            "PMD/beta": float(self.cfg.beta),
            "PMD/entropy_step_coefficient": self.entropy_step_coefficient,
            "PMD/q_coefficient": target.q_coefficient,
            "PMD/candidate_target_kl": target.candidate_kl,
            "PMD/actor_projection_kl": result.mean_kl + result.spread_kl,
            "PMD/actor_mean_kl": result.mean_kl,
            "PMD/actor_spread_kl": result.spread_kl,
            "PMD/effective_sample_size": target.effective_sample_size,
            "PMD/candidate_entropy": target.entropy,
            "PMD/projection_loss_before": loss_before,
            "PMD/projection_loss_after": loss_after,
            "PMD/projection_steps_taken": float(result.steps_taken),
            "PMD/projection_backtracks": float(result.backtracks),
            "Value/q_mean": float(candidates.value.q.mean().item()),
            "Value/reward_q_mean": float(candidates.value.reward_q.mean().item()),
            "Value/entropy_q_mean": float(candidates.value.entropy_q.mean().item()),
            "Value/current_entropy_bonus_mean": float(
                (-self.entropy_step_coefficient * candidates.reference_log_probability)
                .mean()
                .item()
            ),
            **self._collection_policy_metrics(),
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
