"""Recorded demonstrations, replayed as off-policy transitions.

Seeding replay with expert data is what keeps a rare reward from being learned
from a single lucky rollout. On Lift the object was lifted in exactly one of
550 collected rollouts; the critic learned what lifting is worth from that one
sample, and the policy destroyed itself chasing the estimate.

No reward is carried across. Critics score stored *features*, so demonstrations
are relabelled exactly whenever linear reward weights move, which is what an
IRL loop needs from its expert data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from data._hdf5 import iter_hdf5_demos
from storage.observations import (
    Observation,
    detached_observation,
    observation_leaves,
)
from storage.transition_replay import TransitionRollout


def load_demonstration_rollout(
    path: str,
    *,
    feature_names: Sequence[str],
    observation_template: Observation,
    device: torch.device | str,
    max_episodes: int | None = None,
) -> TransitionRollout:
    """Read demonstrations into one ``[E,T-1,...]`` rollout on ``device``.

    Each episode's last step is dropped: its successor observation belongs to
    the next episode, and inventing one would teach the critic a transition the
    simulator never took. ``observation_template`` is a live observation from
    the environment, whose tree shape the loaded observations must match for
    demonstrations and collected data to sample into one batch.
    """
    episodes = []
    for key, demo in iter_hdf5_demos(path, payload_hint="obs/policy"):
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
        episodes.append(_read_episode(key, demo, feature_names))
    if not episodes:
        raise ValueError(f"No demonstrations found in {path!r}.")
    horizons = {episode["actions"].shape[0] for episode in episodes}
    if len(horizons) != 1:
        raise ValueError(
            f"Demonstrations must share one horizon, got {sorted(horizons)}."
        )

    template = detached_observation(observation_template, torch.device(device))
    stacked = {
        name: torch.stack([episode[name] for episode in episodes]).to(device)
        for name in episodes[0]
    }
    observations = stacked.pop("observations")
    next_observations = stacked.pop("next_observations")
    shape = stacked["actions"].shape[:2]
    return TransitionRollout(
        policy_observations=_match_structure(template, observations),
        next_policy_observations=_match_structure(template, next_observations),
        critic_observations=observations,
        next_critic_observations=next_observations,
        terminated=torch.zeros(shape, dtype=torch.bool, device=device),
        truncated=torch.zeros(shape, dtype=torch.bool, device=device),
        bootstrap_valid=torch.ones(shape, dtype=torch.bool, device=device),
        policy_version=0,
        rollout_id=0,
        **stacked,
    )


def _read_episode(
    key: str, demo: Any, feature_names: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Read one demo, dropping the final step that has no stored successor."""
    observations = _as_float(demo["obs"]["policy"])
    actions = _as_float(demo["actions"])
    rewards = _as_float(demo["rewards"]).reshape(-1)
    dones = torch.as_tensor(demo["dones"][:]).reshape(-1).to(torch.bool)
    available = demo["features"]
    missing = [name for name in feature_names if name not in available]
    if missing:
        raise ValueError(f"Demo {key!r} is missing reward features {missing}.")
    features = torch.stack(
        [_as_float(available[name]).reshape(-1) for name in feature_names], dim=-1
    )

    horizon = int(actions.shape[0])
    lengths = {
        "obs/policy": int(observations.shape[0]),
        "rewards": int(rewards.shape[0]),
        "dones": int(dones.shape[0]),
        "features": int(features.shape[0]),
    }
    mismatched = {name: n for name, n in lengths.items() if n != horizon}
    if mismatched:
        raise ValueError(
            f"Demo {key!r} has {horizon} actions but {mismatched} elsewhere."
        )
    if horizon < 2:
        raise ValueError(f"Demo {key!r} is too short to hold a transition.")
    if bool(dones[:-1].any()):
        raise ValueError(
            f"Demo {key!r} ends an episode before its final step, so its stored "
            "successor observations cross a reset."
        )
    return {
        "observations": observations[:-1],
        "next_observations": observations[1:],
        "actions": actions[:-1],
        "features": features[:-1],
        "environment_rewards": rewards[:-1],
    }


def _as_float(dataset: Any) -> torch.Tensor:
    return torch.as_tensor(dataset[:], dtype=torch.float32)


def _match_structure(template: Observation, tensor: torch.Tensor) -> Observation:
    """Place ``tensor`` in the tree shape the environment reports."""
    if isinstance(template, torch.Tensor):
        return tensor
    paths = [path for path, _ in observation_leaves(template)]
    if len(paths) != 1 or "." in paths[0]:
        raise ValueError(
            "Demonstrations store one observation group, but the environment "
            f"reports {paths}; record demonstrations for each group."
        )
    return {paths[0]: tensor}
