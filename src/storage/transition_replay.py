"""Flat transition replay shared by direct-Q and successor-Q learning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

from storage.observations import (
    Observation,
    assert_finite_observation,
    concat_observations,
    map_observation,
    observation_leaves,
)


@dataclass(slots=True)
class TransitionRollout:
    """One rollout chunk stored as explicit ``[N,T,...]`` transitions.

    Shapes:
        policy_observations / next_policy_observations: [N,T,...]
        critic_observations / next_critic_observations: [N,T,O]
        actions: [N,T,A]
        features: [N,T,D]
        environment_rewards, terminated, truncated, bootstrap_valid: [N,T]

    The next observations are stored explicitly rather than shifted out of the
    current ones, so a time-limit truncation can carry the true terminal state
    while the following transition starts from the post-reset state.
    """

    policy_observations: Observation
    next_policy_observations: Observation
    critic_observations: torch.Tensor
    next_critic_observations: torch.Tensor
    actions: torch.Tensor
    features: torch.Tensor
    environment_rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    bootstrap_valid: torch.Tensor
    policy_version: int
    rollout_id: int

    def __post_init__(self) -> None:
        if self.actions.ndim != 3:
            raise ValueError(
                f"Expected actions [N,T,A], got {tuple(self.actions.shape)}."
            )
        num_envs, horizon, _ = self.actions.shape
        nt = (num_envs, horizon)
        if self.features.ndim != 3 or tuple(self.features.shape[:2]) != nt:
            raise ValueError(
                f"Expected features [N,T,D], got {tuple(self.features.shape)}."
            )
        for name in ("critic_observations", "next_critic_observations"):
            tensor = getattr(self, name)
            if tensor.ndim != 3 or tuple(tensor.shape[:2]) != nt:
                raise ValueError(f"Expected {name} [N,T,O], got {tuple(tensor.shape)}.")
        for name in ("policy_observations", "next_policy_observations"):
            for path, leaf in observation_leaves(getattr(self, name)):
                if leaf.ndim < 3 or tuple(leaf.shape[:2]) != nt:
                    raise ValueError(
                        f"Expected {name} '{path}' [N,T,...], got {tuple(leaf.shape)}."
                    )
        for name in (
            "environment_rewards",
            "terminated",
            "truncated",
            "bootstrap_valid",
        ):
            if tuple(getattr(self, name).shape) != nt:
                raise ValueError(
                    f"Expected {name} shape {nt}, "
                    f"got {tuple(getattr(self, name).shape)}."
                )
        for mask in (self.terminated, self.truncated, self.bootstrap_valid):
            if mask.dtype != torch.bool:
                raise TypeError("Termination and bootstrap masks must have dtype bool.")
        if torch.any(self.terminated & self.truncated):
            raise ValueError("A transition cannot be terminated and truncated.")
        if torch.any(self.bootstrap_valid & self.terminated):
            raise ValueError("Terminated transitions must not bootstrap.")
        if self.policy_version < 0 or self.rollout_id < 0:
            raise ValueError("policy_version and rollout_id must be non-negative.")
        self.validate_finite()

    @property
    def num_envs(self) -> int:
        return int(self.actions.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

    @property
    def transition_count(self) -> int:
        return self.num_envs * self.horizon

    def validate_finite(self) -> None:
        assert_finite_observation(self.policy_observations, name="policy observations")
        assert_finite_observation(
            self.next_policy_observations, name="next policy observations"
        )
        tensors = {
            "critic observations": self.critic_observations,
            "next critic observations": self.next_critic_observations,
            "actions": self.actions,
            "features": self.features,
            "environment rewards": self.environment_rewards,
        }
        for name, tensor in tensors.items():
            if not torch.isfinite(tensor).all():
                bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)[:8].tolist()
                raise ValueError(f"Non-finite {name} at indices {bad}.")

    def to_cpu(self) -> "TransitionRollout":
        def copy(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().to("cpu").clone()

        return TransitionRollout(
            policy_observations=map_observation(self.policy_observations, copy),
            next_policy_observations=map_observation(
                self.next_policy_observations, copy
            ),
            critic_observations=copy(self.critic_observations),
            next_critic_observations=copy(self.next_critic_observations),
            actions=copy(self.actions),
            features=copy(self.features),
            environment_rewards=copy(self.environment_rewards),
            terminated=copy(self.terminated),
            truncated=copy(self.truncated),
            bootstrap_valid=copy(self.bootstrap_valid),
            policy_version=self.policy_version,
            rollout_id=self.rollout_id,
        )


@dataclass(slots=True)
class TransitionBatch:
    """Sampled flat transitions.

    Tensor shapes use B for batch, O for critic observations, A for actions,
    and D for features.
    """

    policy_observations: Observation
    next_policy_observations: Observation
    critic_observations: torch.Tensor
    next_critic_observations: torch.Tensor
    actions: torch.Tensor
    features: torch.Tensor
    environment_rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    bootstrap_valid: torch.Tensor


class TransitionReplayBuffer:
    """CPU replay sampled uniformly across recent rollout chunks."""

    def __init__(self, *, capacity_transitions: int) -> None:
        if capacity_transitions <= 0:
            raise ValueError("capacity_transitions must be > 0.")
        self.capacity_transitions = int(capacity_transitions)
        self._rollouts: deque[TransitionRollout] = deque()

    def __len__(self) -> int:
        return len(self._rollouts)

    @property
    def transition_count(self) -> int:
        return sum(rollout.transition_count for rollout in self._rollouts)

    @property
    def policy_versions(self) -> tuple[int, ...]:
        return tuple(rollout.policy_version for rollout in self._rollouts)

    def clear(self) -> None:
        self._rollouts.clear()

    def add(self, rollout: TransitionRollout) -> None:
        stored = rollout.to_cpu()
        if self._rollouts and stored.rollout_id <= self._rollouts[-1].rollout_id:
            raise ValueError("rollout_id must increase strictly.")
        if stored.transition_count > self.capacity_transitions:
            raise ValueError(
                "One rollout contains more transitions than replay capacity: "
                f"{stored.transition_count} > {self.capacity_transitions}."
            )
        self._rollouts.append(stored)
        while self.transition_count > self.capacity_transitions:
            self._rollouts.popleft()

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> TransitionBatch:
        """Sample ``batch_size`` transitions uniformly across stored chunks."""
        if not self._rollouts:
            raise RuntimeError("Cannot sample an empty replay.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        chunk_choices = torch.randint(
            len(self._rollouts), (batch_size,), generator=generator
        )
        pieces: list[TransitionBatch] = []
        for chunk_idx, rollout in enumerate(self._rollouts):
            count = int((chunk_choices == chunk_idx).sum().item())
            if count == 0:
                continue
            flat_index = torch.randint(
                rollout.transition_count, (count,), generator=generator
            )
            # Paired (not Cartesian) selection: sample i takes env_idx[i] at
            # time_idx[i], giving [count, ...] slices of the [N,T,...] chunk.
            env_idx = flat_index // rollout.horizon
            time_idx = flat_index % rollout.horizon
            pieces.append(_gather_transitions(rollout, env_idx, time_idx))
        return _concat_batches(pieces, device)


def _gather_transitions(
    rollout: TransitionRollout, env_idx: torch.Tensor, time_idx: torch.Tensor
) -> TransitionBatch:
    """Gather paired transitions ``(env_idx[i], time_idx[i])`` as ``[B, ...]``."""

    def observation_at(observation: Observation) -> Observation:
        return map_observation(observation, lambda leaf: leaf[env_idx, time_idx])

    return TransitionBatch(
        policy_observations=observation_at(rollout.policy_observations),
        next_policy_observations=observation_at(rollout.next_policy_observations),
        critic_observations=rollout.critic_observations[env_idx, time_idx],
        next_critic_observations=rollout.next_critic_observations[env_idx, time_idx],
        actions=rollout.actions[env_idx, time_idx],
        features=rollout.features[env_idx, time_idx],
        environment_rewards=rollout.environment_rewards[env_idx, time_idx],
        terminated=rollout.terminated[env_idx, time_idx],
        truncated=rollout.truncated[env_idx, time_idx],
        bootstrap_valid=rollout.bootstrap_valid[env_idx, time_idx],
    )


def _concat_batches(
    pieces: list[TransitionBatch], device: torch.device
) -> TransitionBatch:
    """Concatenate per-chunk batches and move the result to ``device``."""
    if not pieces:
        raise RuntimeError("Sampling produced no transitions.")

    def stack(name: str) -> torch.Tensor:
        return torch.cat([getattr(piece, name) for piece in pieces]).to(device)

    return TransitionBatch(
        policy_observations=_to_device_observation(
            concat_observations([piece.policy_observations for piece in pieces]),
            device,
        ),
        next_policy_observations=_to_device_observation(
            concat_observations([piece.next_policy_observations for piece in pieces]),
            device,
        ),
        critic_observations=stack("critic_observations"),
        next_critic_observations=stack("next_critic_observations"),
        actions=stack("actions"),
        features=stack("features"),
        environment_rewards=stack("environment_rewards"),
        terminated=stack("terminated"),
        truncated=stack("truncated"),
        bootstrap_valid=stack("bootstrap_valid"),
    )


def _to_device_observation(
    observation: Observation, device: torch.device
) -> Observation:
    return map_observation(observation, lambda leaf: leaf.to(device))
