"""Flat transition replay shared by direct-Q and successor-Q learning."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields

import torch
from einops import rearrange

from storage.observations import (
    Observation,
    assert_finite_observation,
    concat_observations,
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

    def to_device(self, device: torch.device | str) -> "TransitionRollout":
        """Return a detached copy of this chunk on ``device``."""

        def copy(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.detach().to(device).clone()

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


_OBSERVATION_FIELDS = ("policy_observations", "next_policy_observations")


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


_BATCH_FIELDS = tuple(field.name for field in fields(TransitionBatch))


class TransitionReplayBuffer:
    """Flat ring buffer of transitions, sampled uniformly.

    Storage is allocated once from the first rollout's structure and then
    reused: later inserts overwrite the oldest transitions, so sampling cost is
    independent of how many rollouts have been collected and of how the data
    was chunked. ``device`` decides where the ring lives; keeping it on the
    training device removes a host-to-device copy from every sample.

    Demonstrations, when seeded, are held outside the ring and never evicted,
    and ``demonstration_fraction`` of every sampled batch is drawn from them.
    That is RLPD's mixing rule, and it is what stops a reward the policy only
    reaches by accident from being estimated off a single rollout.
    """

    def __init__(
        self,
        *,
        capacity_transitions: int,
        device: str | torch.device = "cpu",
        demonstration_fraction: float = 0.0,
    ) -> None:
        if capacity_transitions <= 0:
            raise ValueError("capacity_transitions must be > 0.")
        if not 0.0 <= demonstration_fraction < 1.0:
            raise ValueError("demonstration_fraction must lie in [0,1).")
        self.capacity_transitions = int(capacity_transitions)
        self.device = torch.device(device)
        self.demonstration_fraction = float(demonstration_fraction)
        self._demonstrations: TransitionBatch | None = None
        self._storage: TransitionBatch | None = None
        self._policy_versions = torch.empty(0, dtype=torch.int64)
        self._transition_count = 0
        self._cursor = 0
        self._last_rollout_id = -1

    @property
    def transition_count(self) -> int:
        return self._transition_count

    @property
    def demonstration_count(self) -> int:
        """Protected expert transitions; never counted as online interaction."""
        return (
            0
            if self._demonstrations is None
            else int(self._demonstrations.actions.shape[0])
        )

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
        flat = _flatten_rollout(rollout.to_device(self.device))
        if self._storage is None:
            _require_capacity_fits(flat, self.capacity_transitions, self.device)
            self._storage = _allocate_storage(
                flat, self.capacity_transitions, self.device
            )
            self._policy_versions = torch.empty(
                self.capacity_transitions, dtype=torch.int64, device=self.device
            )
        count = rollout.transition_count
        # Contiguous ring write: rows cursor..cursor+count-1, wrapping around.
        rows = (
            self._cursor + torch.arange(count, device=self.device)
        ) % self.capacity_transitions
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
        """Sample ``batch_size`` transitions uniformly with replacement.

        With demonstrations seeded, the draw is split so that the demonstration
        share of the batch is fixed and independent of how much the policy has
        collected. Rows are then shuffled together: contiguous critic-head
        slices must see the same mixture, not different replay partitions.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self._transition_count == 0:
            if self._demonstrations is None:
                raise RuntimeError("Cannot sample an empty replay.")
            # Before online collection, the protected partition is the entire
            # available dataset. This enables demo-only critic initialization.
            return _sample_rows(
                self._demonstrations,
                self.demonstration_count,
                batch_size,
                generator=generator,
                device=device,
                index_device=self.device,
            )
        if self._storage is None:
            raise RuntimeError("Online replay has a count but no storage.")
        demonstration_size = (
            0
            if self._demonstrations is None
            else min(round(batch_size * self.demonstration_fraction), batch_size - 1)
        )
        collected = _sample_rows(
            self._storage,
            self._transition_count,
            batch_size - demonstration_size,
            generator=generator,
            device=device,
            index_device=self.device,
        )
        if demonstration_size == 0:
            return collected
        assert self._demonstrations is not None
        batch = _concat_batches(
            _sample_rows(
                self._demonstrations,
                int(self._demonstrations.actions.shape[0]),
                demonstration_size,
                generator=generator,
                device=device,
                index_device=self.device,
            ),
            collected,
        )
        rows = torch.randperm(batch_size, generator=generator).to(batch.actions.device)
        return _map_batch(batch, lambda leaf: leaf[rows])

    def seed_with_demonstrations(self, rollout: TransitionRollout) -> None:
        """Store demonstrations outside the ring, where nothing evicts them."""
        if self.demonstration_fraction == 0.0:
            raise ValueError("Seeding demonstrations needs demonstration_fraction > 0.")
        if self._demonstrations is not None:
            raise RuntimeError("Demonstrations are already seeded.")
        self._demonstrations = _flatten_rollout(rollout.to_device(self.device))


def _sample_rows(
    storage: TransitionBatch,
    stored: int,
    batch_size: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    index_device: torch.device,
) -> TransitionBatch:
    """Draw ``batch_size`` rows uniformly with replacement from ``stored`` of them."""
    index = torch.randint(stored, (batch_size,), generator=generator).to(index_device)
    return _map_batch(storage, lambda leaf: leaf[index].to(device))


def _concat_batches(*batches: TransitionBatch) -> TransitionBatch:
    """Join sampled batches along the batch dimension."""
    return TransitionBatch(
        **{
            name: (
                concat_observations([getattr(batch, name) for batch in batches])
                if name in _OBSERVATION_FIELDS
                else torch.cat([getattr(batch, name) for batch in batches], dim=0)
            )
            for name in _BATCH_FIELDS
        }
    )


def _flatten_rollout(rollout: TransitionRollout) -> TransitionBatch:
    """Flatten one ``[N,T,...]`` chunk into ``[(N T), ...]`` transitions."""
    return TransitionBatch(
        **{
            name: map_observation(
                getattr(rollout, name),
                lambda leaf: rearrange(leaf, "n t ... -> (n t) ..."),
            )
            for name in _BATCH_FIELDS
        }
    )


def _allocate_storage(
    example: TransitionBatch, capacity: int, device: torch.device
) -> TransitionBatch:
    """Allocate ring storage shaped like one flattened rollout."""
    return _map_batch(
        example,
        lambda leaf: torch.empty(
            (capacity, *leaf.shape[1:]), dtype=leaf.dtype, device=device
        ),
    )


def _write_storage(
    storage: TransitionBatch, flat: TransitionBatch, rows: torch.Tensor
) -> None:
    """Write flattened transitions into the storage rows ``rows`` [M]."""
    for name in _BATCH_FIELDS:
        write_observation(getattr(storage, name), getattr(flat, name), rows)


def _map_batch(
    batch: TransitionBatch, function: Callable[[torch.Tensor], torch.Tensor]
) -> TransitionBatch:
    """Apply ``function`` to every stored tensor, observation leaves included."""
    return TransitionBatch(
        **{
            name: map_observation(getattr(batch, name), function)
            for name in _BATCH_FIELDS
        }
    )


def _batch_tensors(batch: TransitionBatch) -> Iterator[torch.Tensor]:
    """Yield every stored tensor of one batch, observations included."""
    for name in _BATCH_FIELDS:
        for _, leaf in observation_leaves(getattr(batch, name)):
            yield leaf


def _require_capacity_fits(
    example: TransitionBatch, capacity: int, device: torch.device
) -> None:
    """Fail before allocating a ring that does not fit on ``device``."""
    if device.type != "cuda":
        return
    per_transition = sum(
        tensor[0].numel() * tensor.element_size() for tensor in _batch_tensors(example)
    )
    required = per_transition * capacity
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if required >= free_bytes:
        raise ValueError(
            f"Replay of {capacity:,} transitions needs "
            f"{required / 2**30:.1f} GiB on {device} at "
            f"{per_transition} B per transition, but only "
            f"{free_bytes / 2**30:.1f} GiB is free. Lower replay.capacity or "
            "set replay.storage_device to 'cpu'."
        )
