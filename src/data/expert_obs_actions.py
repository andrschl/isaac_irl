"""Load (obs, actions) expert demos from HDF5 files for behavioral cloning."""

from __future__ import annotations

import os
from typing import Any

import torch

from ._hdf5 import iter_hdf5_demos


def _walk_obs_group(group: Any, demo_key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for child_key in sorted(group.keys()):
        child = group[child_key]
        if hasattr(child, "keys"):
            out[str(child_key)] = _walk_obs_group(child, demo_key=demo_key)
        else:
            tensor = torch.as_tensor(child[...])
            if tensor.ndim < 1:
                raise ValueError(
                    f"Demo '{demo_key}' obs leaf '{child_key}' must have shape "
                    f"[T, ...], got scalar."
                )
            out[str(child_key)] = tensor
    return out


def _collect_leaf_lengths(obs: dict[str, Any], out: list[int]) -> None:
    for value in obs.values():
        if isinstance(value, dict):
            _collect_leaf_lengths(value, out)
        else:
            out.append(int(value.shape[0]))


def _extract_obs_action_episodes_from_hdf5(
    path: str,
) -> list[tuple[dict[str, Any], torch.Tensor]]:
    """Load (obs_dict, actions[T, A]) per demo from an HDF5 file written by
    :class:`collectors.RobomimicDataCollector`.

    Expects:
      - ``data/demo_<id>/obs/<group>/<leaf>`` datasets of shape ``[T, ...]``
        (recursively nested groups under ``obs`` become nested dicts).
      - ``data/demo_<id>/actions`` dataset of shape ``[T, A]``.

    Returns one ``(obs_dict, actions)`` tuple per demo. Demos are validated
    for matching ``T`` across all leaves and actions.
    """
    episodes: list[tuple[dict[str, Any], torch.Tensor]] = []
    for demo_key, demo_group in iter_hdf5_demos(path, payload_hint="{obs,actions}"):
        if "obs" not in demo_group:
            raise ValueError(f"Demo '{demo_key}' is missing required group 'obs'.")
        if "actions" not in demo_group:
            raise ValueError(
                f"Demo '{demo_key}' is missing required dataset 'actions'."
            )

        obs_dict = _walk_obs_group(demo_group["obs"], demo_key=demo_key)
        actions_tensor = torch.as_tensor(demo_group["actions"][...])
        if actions_tensor.ndim != 2:
            raise ValueError(
                f"Demo '{demo_key}' actions must have shape [T, A], "
                f"got {tuple(actions_tensor.shape)}."
            )

        lengths: list[int] = []
        _collect_leaf_lengths(obs_dict, lengths)
        t_actions = int(actions_tensor.shape[0])
        if any(length != t_actions for length in lengths):
            raise ValueError(
                f"Demo '{demo_key}' has mismatched T across obs leaves and actions: "
                f"actions T={t_actions}, obs leaf Ts={lengths}."
            )

        episodes.append((obs_dict, actions_tensor))

    if len(episodes) == 0:
        raise ValueError("No demos with (obs, actions) found in HDF5 payload.")

    return episodes


def load_obs_action_episodes(
    path: str,
) -> list[tuple[dict[str, Any], torch.Tensor]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert data path does not exist: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".h5", ".hdf5"}:
        raise ValueError(
            f"Unsupported expert data extension '{ext}' for BC loading. "
            "Only .h5/.hdf5 (written by RobomimicDataCollector) is supported."
        )
    return _extract_obs_action_episodes_from_hdf5(path)
