"""Recorded demonstrations, replayed as off-policy transitions.

Seeding replay with expert data is what keeps a rare reward from being learned
from a single lucky rollout. On Lift the object was lifted in exactly one of
550 collected rollouts; the critic learned what lifting is worth from that one
sample, and the policy destroyed itself chasing the estimate.

No reward is carried across. Critics score stored *features*, so demonstrations
are relabelled exactly whenever linear reward weights move, which is what an
IRL loop needs from its expert data. Nothing else is reconstructed either: the
successor, the two ending flags and the final transition of every episode are
read back exactly as they were recorded, because a bootstrap mask guessed from
a file is a bias no amount of critic capacity notices.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from einops import rearrange

from data.feature_semantics import require_transition_semantics
from storage.observations import (
    Observation,
    critic_observation,
    detached_observation,
    map_observation,
    observation_leaves,
    stack_observations,
)
from storage.transition_replay import TransitionRollout

_REQUIRED_KEYS = ("obs", "next_obs", "actions", "rewards", "terminated", "truncated")


def load_demonstration_rollout(
    path: str,
    *,
    feature_names: Sequence[str],
    observation_template: Observation,
    device: torch.device | str,
    horizon: int,
    step_dt: float,
    is_finite_horizon: bool,
    max_episodes: int | None = None,
) -> TransitionRollout:
    """Read demonstrations into one ``[E,T,...]`` rollout on ``device``.

    ``observation_template`` is a live observation from the environment, whose
    tree shape the loaded observations must match for demonstrations and
    collected data to sample into one batch. ``horizon``, ``step_dt`` and
    ``is_finite_horizon`` are this run's episode definition, checked against the
    one the file was recorded under.
    """
    import h5py

    requested = tuple(str(name) for name in feature_names)
    episodes: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        if data is None:
            raise ValueError(f"No 'data' group in {path!r}.")
        require_transition_semantics(
            data.attrs,
            path=path,
            horizon=horizon,
            step_dt=step_dt,
            is_finite_horizon=is_finite_horizon,
        )
        for key in sorted(data.keys()):
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
            episodes.append(_read_episode(str(key), data[key], requested))
    if not episodes:
        raise ValueError(f"No demonstrations found in {path!r}.")
    lengths = {int(episode["actions"].shape[0]) for episode in episodes}
    if len(lengths) != 1:
        raise ValueError(
            f"Demonstrations must share one episode length, got {sorted(lengths)}."
        )

    device = torch.device(device)
    template = detached_observation(observation_template, device)
    observations = _stack_observations(episodes, "observations", template, path, device)
    next_observations = _stack_observations(
        episodes, "next_observations", template, path, device
    )
    stacked = {
        name: torch.stack([episode[name] for episode in episodes]).to(device)
        for name in ("actions", "features", "environment_rewards")
    }
    terminated = torch.stack([episode["terminated"] for episode in episodes]).to(device)
    truncated = torch.stack([episode["truncated"] for episode in episodes]).to(device)
    return TransitionRollout(
        policy_observations=observations,
        next_policy_observations=next_observations,
        critic_observations=_flat_critic(observations),
        next_critic_observations=_flat_critic(next_observations),
        terminated=terminated,
        truncated=truncated,
        bootstrap_valid=~(terminated | (truncated & bool(is_finite_horizon))),
        policy_version=0,
        rollout_id=0,
        **stacked,
    )


def _stack_observations(
    episodes: list[dict[str, Any]],
    name: str,
    template: Observation,
    path: str,
    device: torch.device,
) -> Observation:
    """Stack one observation tree per episode into ``[E,T,...]`` on ``device``."""
    stacked = stack_observations([episode[name] for episode in episodes], dim=0)
    matched = _match_structure(template, stacked, path=path)
    return map_observation(matched, lambda leaf: leaf.to(device))


def _flat_critic(observations: Observation) -> torch.Tensor:
    """Resolve the critic's view exactly as the online collector does. [E,T,O]."""
    return rearrange(critic_observation(observations), "e t ... -> e t (...)")


def _read_episode(
    key: str, demo: Any, feature_names: tuple[str, ...]
) -> dict[str, Any]:
    """Read one complete episode, keeping every transition it recorded."""
    missing = [name for name in _REQUIRED_KEYS if name not in demo]
    if missing:
        raise ValueError(
            f"Demo {key!r} is missing {missing}. Demonstrations must carry their "
            "own successors and ending flags; re-record them."
        )
    actions = _as_float(demo["actions"])
    horizon = int(actions.shape[0])
    if horizon < 1:
        raise ValueError(f"Demo {key!r} holds no transitions.")
    terminated = _as_bool(demo["terminated"])
    truncated = _as_bool(demo["truncated"])
    features = _episode_features(key, demo, feature_names, horizon=horizon)
    observations = _read_observation_tree(demo["obs"])
    next_observations = _read_observation_tree(demo["next_obs"])
    episode = {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "features": features,
        "environment_rewards": _as_float(demo["rewards"]).reshape(-1),
        "terminated": terminated,
        "truncated": truncated,
    }
    _require_one_ending(key, terminated=terminated, truncated=truncated)
    _require_matching_lengths(key, episode, horizon=horizon)
    return episode


def _episode_features(
    key: str, demo: Any, feature_names: tuple[str, ...], *, horizon: int
) -> torch.Tensor:
    """Stack the requested feature columns. [T, D].

    Extra stored columns are ignored on purpose: one recording serves arms that
    learn over different subsets of it, and re-recording per subset would make
    them different data. A *missing* column is an error - it cannot be filled in.
    """
    available = demo.get("features")
    if available is None:
        raise ValueError(f"Demo {key!r} has no 'features' group.")
    missing = [name for name in feature_names if name not in available]
    if missing:
        raise ValueError(
            f"Demo {key!r} is missing reward features {missing}; it holds "
            f"{sorted(str(name) for name in available.keys())}."
        )
    columns = [_as_float(available[name]).reshape(-1) for name in feature_names]
    for name, column in zip(feature_names, columns, strict=True):
        if int(column.shape[0]) != horizon:
            raise ValueError(
                f"Demo {key!r} feature {name!r} has {int(column.shape[0])} steps, "
                f"but {horizon} actions."
            )
    return torch.stack(columns, dim=-1)


def _require_one_ending(
    key: str, *, terminated: torch.Tensor, truncated: torch.Tensor
) -> None:
    """A stored episode ends once, on its last step, or its successors lie."""
    ended = terminated | truncated
    if not bool(ended[-1]):
        raise ValueError(
            f"Demo {key!r} does not end on its last step, so it is a fragment "
            "rather than a complete episode."
        )
    if bool(ended[:-1].any()):
        raise ValueError(
            f"Demo {key!r} ends an episode before its final step, so its stored "
            "successor observations cross a reset."
        )


def _require_matching_lengths(
    key: str, episode: dict[str, Any], *, horizon: int
) -> None:
    lengths: dict[str, int] = {}
    for name, value in episode.items():
        if isinstance(value, torch.Tensor):
            lengths[name] = int(value.shape[0])
            continue
        for path, leaf in observation_leaves(value):
            lengths[f"{name}/{path}"] = int(leaf.shape[0])
    mismatched = {name: n for name, n in lengths.items() if n != horizon}
    if mismatched:
        raise ValueError(
            f"Demo {key!r} has {horizon} actions but {mismatched} elsewhere."
        )


def _read_observation_tree(group: Any) -> Observation:
    """Read a stored observation group into the nested tensor type. [T, ...]."""
    if not hasattr(group, "keys"):
        return _as_float(group)
    return {
        str(key): _read_observation_tree(group[key]) for key in sorted(group.keys())
    }


def _as_float(dataset: Any) -> torch.Tensor:
    return torch.as_tensor(dataset[:], dtype=torch.float32)


def _as_bool(dataset: Any) -> torch.Tensor:
    return torch.as_tensor(dataset[:]).reshape(-1).to(torch.bool)


def _match_structure(
    template: Observation, observations: Observation, *, path: str
) -> Observation:
    """Check stored observations against the tree the environment reports."""
    stored_paths = sorted(name for name, _ in observation_leaves(observations))
    template_paths = sorted(name for name, _ in observation_leaves(template))
    if isinstance(template, torch.Tensor) and len(stored_paths) == 1:
        # A bare-tensor environment stores its single group under a name.
        leaf = next(leaf for _, leaf in observation_leaves(observations))
        return leaf
    if stored_paths != template_paths:
        raise ValueError(
            f"Demonstrations at {path!r} store observation groups {stored_paths}, "
            f"but the environment reports {template_paths}; re-record them "
            "against this observation view."
        )
    return observations
