"""Flat transition replay shared by direct-Q and successor-Q learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange

from storage.observations import (
    Observation,
    assert_finite_observation,
    empty_observation_like,
    map_observation,
    observation_leaves,
    write_observation,
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
    """Flat CPU ring buffer of transitions, sampled uniformly.

    Storage is allocated once from the first rollout's structure and then
    reused: later inserts overwrite the oldest transitions, so sampling cost is
    independent of how many rollouts have been collected and of how the data
    was chunked.
    """

    def __init__(self, *, capacity_transitions: int) -> None:
        if capacity_transitions <= 0:
            raise ValueError("capacity_transitions must be > 0.")
        self.capacity_transitions = int(capacity_transitions)
        self._storage: TransitionBatch | None = None
        self._policy_versions = torch.empty(0, dtype=torch.int64)
        self._transition_count = 0
        self._cursor = 0
        self._last_rollout_id = -1

    @property
    def transition_count(self) -> int:
        return self._transition_count

    def policy_version_range(self) -> tuple[int, int]:
        """Oldest and newest policy version currently stored."""
        if self._transition_count == 0:
            raise RuntimeError("Replay is empty.")
        stored = self._policy_versions[: self._transition_count]
        return int(stored.min().item()), int(stored.max().item())

    def clear(self) -> None:
        self._transition_count = 0
        self._cursor = 0

    def add(self, rollout: TransitionRollout) -> None:
        """Append one rollout's transitions, overwriting the oldest if needed."""
        if rollout.rollout_id <= self._last_rollout_id:
            raise ValueError("rollout_id must increase strictly.")
        if rollout.transition_count > self.capacity_transitions:
            raise ValueError(
                "One rollout contains more transitions than replay capacity: "
                f"{rollout.transition_count} > {self.capacity_transitions}."
            )
        flat = _flatten_rollout(rollout.to_cpu())
        if self._storage is None:
            self._storage = _allocate_storage(flat, self.capacity_transitions)
            self._policy_versions = torch.empty(
                self.capacity_transitions, dtype=torch.int64
            )
        count = rollout.transition_count
        # Contiguous ring write: rows cursor..cursor+count-1, wrapping around.
        rows = (self._cursor + torch.arange(count)) % self.capacity_transitions
        _write_storage(self._storage, flat, rows)
        self._policy_versions[rows] = rollout.policy_version
        self._cursor = int((self._cursor + count) % self.capacity_transitions)
        self._transition_count = min(
            self._transition_count + count, self.capacity_transitions
        )
        self._last_rollout_id = rollout.rollout_id

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> TransitionBatch:
        """Sample ``batch_size`` stored transitions uniformly with replacement."""
        if self._storage is None or self._transition_count == 0:
            raise RuntimeError("Cannot sample an empty replay.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        index = torch.randint(
            self._transition_count, (batch_size,), generator=generator
        )
        return _gather_transitions(self._storage, index, device)


def _flatten_rollout(rollout: TransitionRollout) -> TransitionBatch:
    """Flatten one ``[N,T,...]`` chunk into ``[(N T), ...]`` transitions."""

    def flatten(tensor: torch.Tensor) -> torch.Tensor:
        return rearrange(tensor, "n t ... -> (n t) ...")

    return TransitionBatch(
        policy_observations=map_observation(rollout.policy_observations, flatten),
        next_policy_observations=map_observation(
            rollout.next_policy_observations, flatten
        ),
        critic_observations=flatten(rollout.critic_observations),
        next_critic_observations=flatten(rollout.next_critic_observations),
        actions=flatten(rollout.actions),
        features=flatten(rollout.features),
        environment_rewards=flatten(rollout.environment_rewards),
        terminated=flatten(rollout.terminated),
        truncated=flatten(rollout.truncated),
        bootstrap_valid=flatten(rollout.bootstrap_valid),
    )


def _allocate_storage(example: TransitionBatch, capacity: int) -> TransitionBatch:
    """Allocate ring storage shaped like one flattened rollout."""

    def allocate(tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty((capacity, *tensor.shape[1:]), dtype=tensor.dtype)

    return TransitionBatch(
        policy_observations=empty_observation_like(
            example.policy_observations, capacity
        ),
        next_policy_observations=empty_observation_like(
            example.next_policy_observations, capacity
        ),
        critic_observations=allocate(example.critic_observations),
        next_critic_observations=allocate(example.next_critic_observations),
        actions=allocate(example.actions),
        features=allocate(example.features),
        environment_rewards=allocate(example.environment_rewards),
        terminated=allocate(example.terminated),
        truncated=allocate(example.truncated),
        bootstrap_valid=allocate(example.bootstrap_valid),
    )


def _write_storage(
    storage: TransitionBatch, flat: TransitionBatch, rows: torch.Tensor
) -> None:
    """Write flattened transitions into the storage rows ``rows`` [M]."""
    write_observation(storage.policy_observations, flat.policy_observations, rows)
    write_observation(
        storage.next_policy_observations, flat.next_policy_observations, rows
    )
    for name in (
        "critic_observations",
        "next_critic_observations",
        "actions",
        "features",
        "environment_rewards",
        "terminated",
        "truncated",
        "bootstrap_valid",
    ):
        getattr(storage, name)[rows] = getattr(flat, name)


def _gather_transitions(
    storage: TransitionBatch, index: torch.Tensor, device: torch.device
) -> TransitionBatch:
    """Gather stored rows ``index`` [B] and move them to ``device``."""

    def take(name: str) -> torch.Tensor:
        return getattr(storage, name)[index].to(device)

    def take_observation(observation: Observation) -> Observation:
        return map_observation(observation, lambda leaf: leaf[index].to(device))

    return TransitionBatch(
        policy_observations=take_observation(storage.policy_observations),
        next_policy_observations=take_observation(storage.next_policy_observations),
        critic_observations=take("critic_observations"),
        next_critic_observations=take("next_critic_observations"),
        actions=take("actions"),
        features=take("features"),
        environment_rewards=take("environment_rewards"),
        terminated=take("terminated"),
        truncated=take("truncated"),
        bootstrap_valid=take("bootstrap_valid"),
    )
