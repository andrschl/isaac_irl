"""Shared state-dependent tanh Gaussian actor for SAC and PMD."""

from __future__ import annotations

import copy
import math
from typing import Any, Protocol

import torch
import torch.nn as nn

from storage.observations import actor_observation


def policy_observation_tensor(observations: Any) -> torch.Tensor:
    """Resolve policy observations to a flat tensor ``[B,O]``."""
    return actor_observation(observations).flatten(start_dim=1)


class StateDependentTanhGaussianActor(nn.Module):
    """Map normalized observations ``[B,O]`` to bounded Gaussian actions ``[B,A]``."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        action_bound: float = 1.0,
        minimum_log_std: float = -20.0,
        initial_std: float = 1.0,
        maximum_log_std: float = 2.0,
        log_probability_epsilon: float = 1.0e-6,
        observation_normalizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim) <= 0:
            raise ValueError("observation_dim and action_dim must be positive.")
        if not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive dimensions.")
        if action_bound <= 0.0:
            raise ValueError("action_bound must be > 0.")
        if minimum_log_std >= maximum_log_std:
            raise ValueError("minimum_log_std must be below maximum_log_std.")
        if initial_std <= 0.0:
            raise ValueError("initial_std must be > 0.")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.action_bound = float(action_bound)
        self.minimum_log_std = float(minimum_log_std)
        self.maximum_log_std = float(maximum_log_std)
        self.log_probability_epsilon = float(log_probability_epsilon)
        self.observation_normalizer = observation_normalizer or nn.Identity()

        dimensions = (observation_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:], strict=True):
            layers.extend((nn.Linear(input_dim, output_dim), nn.ELU()))
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(dimensions[-1], action_dim)
        self.log_std_head = nn.Linear(dimensions[-1], action_dim)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, math.log(initial_std))

    def distribution_parameters(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent Gaussian mean and std, both ``[B,A]``."""
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(
                f"Expected observations [B,{self.observation_dim}], "
                f"got {tuple(observations.shape)}."
            )
        normalized = self.observation_normalizer(observations)
        latent = self.trunk(normalized)
        mean = self.mean_head(latent)
        log_std = self.log_std_head(latent).clamp(
            self.minimum_log_std, self.maximum_log_std
        )
        return mean, log_std.exp()

    def squash(self, latent_actions: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent_actions) * self.action_bound

    def corrected_log_probability(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return corrected bounded-action log probabilities ``[B]``."""
        distribution = torch.distributions.Normal(mean, std)
        normalized_actions = torch.tanh(latent_actions)
        correction = math.log(self.action_bound) + torch.log(
            1.0 - normalized_actions.square() + self.log_probability_epsilon
        )
        return (distribution.log_prob(latent_actions) - correction).sum(dim=-1)


class StateDependentTanhPolicy:
    """Observation-aware policy facade shared by SAC and exponential PMD."""

    def __init__(self, model: StateDependentTanhGaussianActor) -> None:
        self.model = model

    def _parameters(self, observations: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.distribution_parameters(
            policy_observation_tensor(observations)
        )

    def _sample(
        self,
        observations: Any,
        *,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return bounded actions [B,A], latent actions [B,A], log probs [B]."""
        mean, std = self._parameters(observations)
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        latent_actions = mean + std * noise
        actions = self.model.squash(latent_actions)
        log_probability = self.model.corrected_log_probability(
            mean, std, latent_actions
        )
        return actions, latent_actions, log_probability

    def sample_with_log_probability(
        self,
        observations: Any,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample bounded actions and corrected log probabilities."""
        actions, _, log_probability = self._sample(observations, generator=generator)
        return actions, log_probability

    def sample_with_latent(
        self,
        observations: Any,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample and also return the pre-squash latents that produced them.

        Keeping the latents avoids inverting tanh later: candidates near the
        action bound cannot be recovered accurately by ``atanh``.
        """
        return self._sample(observations, generator=generator)

    def rsample_with_log_probability(
        self,
        observations: Any,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized bounded actions and corrected log probabilities."""
        actions, _, log_probability = self._sample(observations, generator=generator)
        return actions, log_probability

    def log_probability(self, observations: Any, actions: torch.Tensor) -> torch.Tensor:
        """Log density of externally supplied bounded actions [B,A] -> [B].

        Inverting tanh saturates near the bound; prefer
        :meth:`log_probability_from_latent` whenever the latents are known.
        """
        mean, std = self._parameters(observations)
        normalized = (actions / self.model.action_bound).clamp(-0.999999, 0.999999)
        latent_actions = torch.atanh(normalized)
        return self.model.corrected_log_probability(mean, std, latent_actions)

    def log_probability_from_latent(
        self, observations: Any, latent_actions: torch.Tensor
    ) -> torch.Tensor:
        """Exact log density of latent actions [B,A] under this policy -> [B]."""
        mean, std = self._parameters(observations)
        return self.model.corrected_log_probability(mean, std, latent_actions)

    def deterministic_action(self, observations: Any) -> torch.Tensor:
        mean, _ = self._parameters(observations)
        return self.model.squash(mean)

    def distribution_parameters(
        self, observations: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._parameters(observations)

    def snapshot(self) -> "StateDependentTanhPolicy":
        """Freeze the network weights while sharing the live normalizer.

        The observation normalizer keeps absorbing new rollouts, and a frozen
        copy of it would make the snapshot normalize with stale moments: its
        actions and its KL from the live policy would then move whenever the
        moments move, not only when the weights do.
        """
        frozen_model = copy.deepcopy(self.model)
        frozen_model.eval()
        frozen_model.requires_grad_(False)
        frozen_model.observation_normalizer = self.model.observation_normalizer
        return StateDependentTanhPolicy(frozen_model)

    def act(self, observations: Any) -> torch.Tensor:
        return self.sample_with_log_probability(observations)[0]

    def act_inference(self, observations: Any) -> torch.Tensor:
        return self.deterministic_action(observations)

    def parameters(self):
        return self.model.parameters()

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()


class TanhPolicyProtocol(Protocol):
    """Bounded policy interface used by off-policy actor/value learners."""

    model: nn.Module

    def sample_with_log_probability(
        self, observations: Any, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def sample_with_latent(
        self, observations: Any, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def rsample_with_log_probability(
        self, observations: Any, *, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def log_probability(
        self, observations: Any, actions: torch.Tensor
    ) -> torch.Tensor: ...

    def log_probability_from_latent(
        self, observations: Any, latent_actions: torch.Tensor
    ) -> torch.Tensor: ...

    def deterministic_action(self, observations: Any) -> torch.Tensor: ...

    def act(self, observations: Any) -> torch.Tensor: ...

    def act_inference(self, observations: Any) -> torch.Tensor: ...

    def distribution_parameters(
        self, observations: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def snapshot(self) -> "TanhPolicyProtocol": ...

    def parameters(self): ...

    def train(self) -> None: ...
