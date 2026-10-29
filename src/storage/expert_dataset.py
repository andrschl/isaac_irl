"""The expert (observation, action) pairs behavioral cloning trains on."""

from __future__ import annotations

import torch

from storage.observations import (
    Observation,
    map_observation,
    observation_leaves,
)


class ExpertObsActionDataset:
    """Flat expert transitions, loaded once and sampled many times.

    Episode boundaries are dropped deliberately: BC is a supervised loss over
    individual (observation, action) pairs and gains nothing from padding or
    masking. Observations keep the nested group structure the actor consumes;
    every leaf is flattened to ``[N_total, *leaf_shape]`` so a minibatch is one
    ``randint`` and one index per leaf.
    """

    def __init__(
        self,
        episodes: list[tuple[Observation, torch.Tensor]],
        *,
        store_device: str | torch.device = "cpu",
        store_dtype: torch.dtype = torch.float32,
    ) -> None:
        if not episodes:
            raise ValueError("Cannot build an expert dataset from zero episodes.")
        device = torch.device(store_device)

        first_observations, first_actions = episodes[0]
        if first_actions.ndim != 2:
            raise ValueError(
                f"Expected actions [T, A], got {tuple(first_actions.shape)}."
            )
        self.action_dim = int(first_actions.shape[-1])
        structure = _leaf_paths(first_observations)

        for index, (observations, actions) in enumerate(episodes):
            if _leaf_paths(observations) != structure:
                raise ValueError(
                    f"Episode {index} has observation leaves {_leaf_paths(observations)}, "
                    f"expected {structure}."
                )
            if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Episode {index} actions must be [T, {self.action_dim}], got "
                    f"{tuple(actions.shape)}."
                )
            if any(
                leaf.shape[0] != actions.shape[0]
                for _, leaf in observation_leaves(observations)
            ):
                raise ValueError(
                    f"Episode {index} observations and actions disagree on length."
                )

        self._actions = torch.cat(
            [
                actions.detach().to(device=device, dtype=store_dtype)
                for _, actions in episodes
            ]
        )
        self._observations = _concat_episode_observations(
            [observations for observations, _ in episodes],
            device=device,
            dtype=store_dtype,
        )
        self._store_device = device

    def __len__(self) -> int:
        return int(self._actions.shape[0])

    def all_pairs(self, device: torch.device | str) -> tuple[Observation, torch.Tensor]:
        """Every stored pair, for a loss that should not be a random draw."""
        device = torch.device(device)
        return (
            map_observation(self._observations, lambda leaf: leaf.to(device)),
            self._actions.to(device),
        )

    def sample(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[Observation, torch.Tensor]:
        """Draw ``batch_size`` pairs with replacement. Shapes: obs [B,...], actions [B,A]."""
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be > 0, got {batch_size}.")
        device = torch.device(device)
        index = torch.randint(0, len(self), (batch_size,), device=self._store_device)
        observations = map_observation(
            self._observations, lambda leaf: leaf.index_select(0, index).to(device)
        )
        return observations, self._actions.index_select(0, index).to(device)


def _leaf_paths(observations: Observation) -> tuple[str, ...]:
    return tuple(path for path, _ in observation_leaves(observations))


def _concat_episode_observations(
    episodes: list[Observation], *, device: torch.device, dtype: torch.dtype
) -> Observation:
    """Concatenate matching observation trees along their time axis."""
    if len(episodes) == 1:
        return map_observation(
            episodes[0], lambda leaf: leaf.detach().to(device=device, dtype=dtype)
        )
    if isinstance(episodes[0], torch.Tensor):
        return torch.cat(
            [episode.detach().to(device=device, dtype=dtype) for episode in episodes]
        )
    return {
        key: _concat_episode_observations(
            [episode[key] for episode in episodes], device=device, dtype=dtype
        )
        for key in episodes[0]
    }
