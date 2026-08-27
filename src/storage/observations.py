"""Generic nested observation operations shared by storage and adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeAlias

import torch

Observation: TypeAlias = torch.Tensor | dict[str, "Observation"]


def observation_leaves(
    observation: Observation,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield path/tensor leaves from a nested observation."""
    if isinstance(observation, torch.Tensor):
        yield "<root>", observation
        return
    if not isinstance(observation, Mapping):
        raise TypeError(
            f"Expected tensor or nested mapping, got {type(observation)!r}."
        )
    for key, value in observation.items():
        for child_path, tensor in observation_leaves(value):
            path = str(key) if child_path == "<root>" else f"{key}.{child_path}"
            yield path, tensor


def map_observation(
    observation: Observation,
    function: Callable[[torch.Tensor], torch.Tensor],
) -> Observation:
    """Apply function independently to every tensor leaf."""
    if isinstance(observation, torch.Tensor):
        return function(observation)
    return {key: map_observation(value, function) for key, value in observation.items()}


def detached_observation(observation: Any, device: torch.device) -> Observation:
    """Detach an environment observation into the core nested-tensor type."""
    if isinstance(observation, torch.Tensor):
        return observation.detach().to(device).clone()
    if isinstance(observation, Mapping):
        return {
            str(key): detached_observation(value, device)
            for key, value in observation.items()
        }
    raise TypeError(
        f"Expected tensor or nested observation mapping, got {type(observation)!r}."
    )


def where_observation(
    mask: torch.Tensor, chosen: Observation, otherwise: Observation
) -> Observation:
    """Select leaves from ``chosen`` where ``mask`` [N] is true, else ``otherwise``.

    The mask indexes the leading environment dimension of every leaf.
    """
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError(f"Expected a bool mask [N], got {tuple(mask.shape)}.")
    if isinstance(chosen, torch.Tensor):
        if not isinstance(otherwise, torch.Tensor):
            raise ValueError("Observation structures must match.")
        if chosen.shape != otherwise.shape:
            raise ValueError(
                "Observation leaves must have matching shapes, got "
                f"{tuple(chosen.shape)} and {tuple(otherwise.shape)}."
            )
        # Broadcast the [N] mask over the trailing leaf dimensions.
        leaf_mask = mask.reshape(-1, *([1] * (chosen.ndim - 1)))
        return torch.where(leaf_mask, chosen, otherwise)
    if not isinstance(otherwise, Mapping) or tuple(chosen) != tuple(otherwise):
        raise ValueError("Observation structures must match.")
    return {key: where_observation(mask, chosen[key], otherwise[key]) for key in chosen}


def select_observation_group(
    observation: Observation, preferred_keys: tuple[str, ...]
) -> torch.Tensor:
    """Resolve a nested observation to one tensor with a single, stated rule.

    The first present group in ``preferred_keys`` wins; otherwise a lone leaf is
    returned, and several leaves are concatenated along their last dimension.
    Leading dimensions are preserved, so callers decide when to flatten.
    """
    if isinstance(observation, torch.Tensor):
        return observation
    if not isinstance(observation, Mapping):
        raise TypeError(
            f"Expected tensor or nested mapping, got {type(observation)!r}."
        )
    for key in preferred_keys:
        value = observation.get(key)
        if isinstance(value, torch.Tensor):
            return value
    leaves = [tensor for _, tensor in observation_leaves(observation)]
    if not leaves:
        raise ValueError(
            "Could not resolve observations: no tensor leaf and none of the "
            f"groups {preferred_keys} is present."
        )
    if len(leaves) == 1:
        return leaves[0]
    return torch.cat(leaves, dim=-1)


def actor_observation(observation: Observation) -> torch.Tensor:
    """Resolve the tensor an actor consumes, preferring the actor/policy group."""
    return select_observation_group(observation, ("actor", "policy"))


def critic_observation(observation: Observation) -> torch.Tensor:
    """Resolve the tensor a critic consumes, preferring the critic group."""
    return select_observation_group(observation, ("critic", "policy", "actor"))


def concat_observations(observations: list[Observation]) -> Observation:
    """Concatenate matching observation trees along their leading dimension."""
    if not observations:
        raise ValueError("Cannot concatenate an empty observation list.")
    first = observations[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(item, torch.Tensor) for item in observations):
            raise ValueError("Observation structures must match across chunks.")
        return torch.cat(observations)
    keys = tuple(first)
    if any(
        not isinstance(item, Mapping) or tuple(item) != keys for item in observations
    ):
        raise ValueError("Observation structures must match across chunks.")
    return {
        key: concat_observations([item[key] for item in observations]) for key in keys
    }


def stack_observations(observations: list[Observation], dim: int) -> Observation:
    if not observations:
        raise ValueError("Cannot stack an empty observation list.")
    first = observations[0]
    if isinstance(first, torch.Tensor):
        if not all(isinstance(item, torch.Tensor) for item in observations):
            raise ValueError("Observation structure changed during collection.")
        return torch.stack(observations, dim=dim)
    keys = tuple(first)
    if any(
        not isinstance(item, Mapping) or tuple(item) != keys for item in observations
    ):
        raise ValueError("Observation structure changed during collection.")
    return {
        key: stack_observations([item[key] for item in observations], dim)
        for key in keys
    }


def assert_finite_observation(observation: Observation, *, name: str) -> None:
    for path, tensor in observation_leaves(observation):
        if not torch.isfinite(tensor).all():
            bad_indices = (~torch.isfinite(tensor)).nonzero(as_tuple=False)[:8].tolist()
            raise ValueError(
                f"Non-finite {name} leaf '{path}' at indices {bad_indices}."
            )
