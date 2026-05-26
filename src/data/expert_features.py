"""Load expert *feature* trajectories from disk.

Supports two payload formats:

  - ``.pt`` / ``.pth`` torch payloads: a tensor ``[T, D]`` / ``[E, T, D]`` or
    a dict carrying one under ``episodes`` / ``features`` / ``expert_episodes``.
  - ``.h5`` / ``.hdf5`` files written by
    :class:`collectors.RobomimicDataCollector`: each demo holds a
    ``features/<feature_name>`` group of shape-``[T]`` (or ``[T, 1]``)
    datasets. Loaded features preserve the HDF5 group order, which matches
    the online reward-manager feature ordering.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from einops import rearrange

from ._hdf5 import iter_hdf5_demos


def _torch_load_payload(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        pass
    except Exception:
        pass
    return torch.load(path, map_location="cpu")


def _tensor_to_feature_episodes(payload: torch.Tensor) -> list[torch.Tensor]:
    if payload.ndim == 2:
        return [payload]
    if payload.ndim == 3:
        return [payload[i] for i in range(payload.shape[0])]
    raise ValueError(
        f"Expected tensor payload with shape [T, D] or [E, T, D], got {tuple(payload.shape)}"
    )


def _extract_feature_episodes_from_torch_payload(payload: Any) -> list[torch.Tensor]:
    if isinstance(payload, dict):
        keys = ("episodes", "features", "expert_episodes")
        selected_key = next((key for key in keys if key in payload), None)
        if selected_key is None:
            raise ValueError(
                "Torch payload must contain one of keys: 'episodes', 'features', 'expert_episodes'."
            )
        payload = payload[selected_key]

    if isinstance(payload, torch.Tensor):
        return _tensor_to_feature_episodes(payload)

    if not isinstance(payload, (list, tuple)):
        raise TypeError(f"Expected list/tuple/tensor feature payload, got {type(payload)!r}")

    episodes: list[torch.Tensor] = []
    for episode in payload:
        episode_tensor = episode if isinstance(episode, torch.Tensor) else torch.as_tensor(episode)
        episodes.extend(_tensor_to_feature_episodes(episode_tensor))
    return episodes


def _extract_feature_episodes_from_hdf5(
    path: str,
    expected_feature_names: list[str] | None = None,
) -> list[torch.Tensor]:
    def _episode_from_named_features(feature_group: Any, *, demo_key: str) -> torch.Tensor:
        if not hasattr(feature_group, "keys"):
            raise ValueError(
                f"Demo '{demo_key}' uses legacy feature format. Expected group 'features/<feature_name>'. "
                "Regenerate demos with the simplified recorder."
            )

        available_feature_names = [str(name) for name in feature_group.keys()]
        if len(available_feature_names) == 0:
            raise ValueError(f"Demo '{demo_key}' has empty 'features' group.")

        if expected_feature_names is None:
            # Preserve the HDF5 group order. Demo files written by
            # RobomimicDataCollector use track_order=True, so this matches the
            # reward-manager feature order used online.
            feature_names = available_feature_names
        else:
            expected_names = [str(name) for name in expected_feature_names]
            available_set = set(available_feature_names)
            expected_set = set(expected_names)
            if available_set != expected_set:
                missing = sorted(expected_set - available_set)
                extra = sorted(available_set - expected_set)
                raise ValueError(
                    f"Demo '{demo_key}' feature-name mismatch: missing={missing}, extra={extra}. "
                    "Regenerate demos with the current feature map, or pass matching ignored/forced terms."
                )
            feature_names = expected_names

        columns: list[torch.Tensor] = []
        expected_steps: int | None = None
        for feature_name in feature_names:
            feature_node = feature_group[feature_name]
            if hasattr(feature_node, "keys"):
                raise ValueError(
                    f"Demo '{demo_key}' feature '{feature_name}' must be a dataset of shape [T] or [T,1]."
                )

            values = torch.as_tensor(feature_node[...])
            if values.ndim == 2 and int(values.shape[1]) == 1:
                values = rearrange(values, "t 1 -> t")
            elif values.ndim != 1:
                raise ValueError(
                    f"Demo '{demo_key}' feature '{feature_name}' must have shape [T] or [T,1], "
                    f"got {tuple(values.shape)}."
                )

            num_steps = int(values.shape[0])
            if expected_steps is None:
                if num_steps <= 0:
                    raise ValueError(f"Demo '{demo_key}' has empty feature sequence '{feature_name}'.")
                expected_steps = num_steps
            elif num_steps != expected_steps:
                raise ValueError(
                    f"Demo '{demo_key}' feature length mismatch: expected {expected_steps}, "
                    f"feature '{feature_name}' has length {num_steps}."
                )

            columns.append(rearrange(values, "t -> t 1"))

        return torch.cat(columns, dim=1)

    episodes: list[torch.Tensor] = []
    for demo_key, demo_group in iter_hdf5_demos(path, payload_hint="features/<feature_name>"):
        if "features" not in demo_group:
            raise ValueError(f"Demo '{demo_key}' is missing required group 'features'.")
        episodes.append(_episode_from_named_features(demo_group["features"], demo_key=demo_key))

    if len(episodes) == 0:
        raise ValueError("No demos found under 'data/*' in HDF5 payload.")

    return episodes


def _validate_feature_episodes(
    episodes: list[torch.Tensor],
    expected_feature_dim: int,
) -> list[torch.Tensor]:
    if len(episodes) == 0:
        raise ValueError("No expert feature episodes found.")

    validated: list[torch.Tensor] = []
    for episode_idx, episode in enumerate(episodes):
        episode_tensor = episode if isinstance(episode, torch.Tensor) else torch.as_tensor(episode)
        if episode_tensor.ndim != 2:
            raise ValueError(
                f"Episode {episode_idx} must have shape [T, D], got {tuple(episode_tensor.shape)}"
            )
        if episode_tensor.shape[0] <= 0:
            raise ValueError(f"Episode {episode_idx} is empty. Expected at least one timestep.")
        if int(episode_tensor.shape[1]) != int(expected_feature_dim):
            raise ValueError(
                f"Episode {episode_idx} feature dim mismatch: "
                f"expected {expected_feature_dim}, got {int(episode_tensor.shape[1])}."
            )
        validated.append(episode_tensor)

    return validated


def load_feature_episodes(
    path: str,
    expected_feature_dim: int,
    expected_feature_names: list[str] | None = None,
) -> list[torch.Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert data path does not exist: {path}")
    if expected_feature_names is not None and len(expected_feature_names) != int(expected_feature_dim):
        raise ValueError(
            f"Expected {expected_feature_dim} feature names, got {len(expected_feature_names)}."
        )

    ext = os.path.splitext(path)[1].lower()
    if ext in {".h5", ".hdf5"}:
        episodes = _extract_feature_episodes_from_hdf5(
            path, expected_feature_names=expected_feature_names
        )
    elif ext in {".pt", ".pth"}:
        payload = _torch_load_payload(path)
        episodes = _extract_feature_episodes_from_torch_payload(payload)
    else:
        raise ValueError(
            f"Unsupported expert data extension '{ext}'. Expected one of: .pt, .pth, .h5, .hdf5"
        )

    return _validate_feature_episodes(episodes, expected_feature_dim=expected_feature_dim)


def load_expert_success_rate(path: str) -> float | None:
    """Mean any-time success over expert demos.

    Reads the per-step ``success`` dataset stored per demo at recording time
    (``data/demo_*/success`` of shape ``[T]``); a demo counts as a success if it
    reached the goal at any step. Returns None when the file carries no success
    info — e.g. demos recorded before the success metric existed, or non-HDF5
    payloads — so the caller can disable the expert-success gap gracefully.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".h5", ".hdf5"}:
        return None

    per_demo_success: list[float] = []
    for _demo_key, demo_group in iter_hdf5_demos(path, payload_hint="success"):
        if "success" not in demo_group:
            continue
        values = torch.as_tensor(demo_group["success"][...]).reshape(-1)
        if values.numel() == 0:
            continue
        per_demo_success.append(float((values > 0.5).any().item()))

    if not per_demo_success:
        return None
    return sum(per_demo_success) / len(per_demo_success)
