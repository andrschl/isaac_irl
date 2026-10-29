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
from typing import TYPE_CHECKING, Any

import torch
from einops import rearrange

from ._hdf5 import iter_hdf5_demos
from .feature_semantics import (
    FEATURE_CAPTURE_TIMING,
    FEATURE_SEMANTICS_VERSION,
    FEATURE_VALUE_SEMANTICS,
)
from .sampling import subset_episodes

if TYPE_CHECKING:
    from algorithms.irl import IrlAlgorithm


def _validate_hdf5_feature_semantics(path: str) -> None:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Loading HDF5 demonstrations requires h5py.") from exc

    expected = {
        "feature_semantics_version": FEATURE_SEMANTICS_VERSION,
        "feature_capture_timing": FEATURE_CAPTURE_TIMING,
        "feature_value_semantics": FEATURE_VALUE_SEMANTICS,
    }
    with h5py.File(path, "r") as h5_file:
        if "data" not in h5_file:
            raise ValueError("HDF5 demonstration file is missing group 'data'.")
        metadata = h5_file["data"].attrs
        mismatches = {
            key: metadata.get(key)
            for key, expected_value in expected.items()
            if metadata.get(key) != expected_value
        }
    if mismatches:
        raise ValueError(
            "Demonstrations use missing or legacy reward-feature semantics "
            f"{mismatches}. Regenerate them with scripts/record_demos.py."
        )


def _torch_load_payload(path: str) -> Any:
    """Load a .pt demonstration file as tensors only.

    ``weights_only=True`` is not a performance detail: the fallback it replaces
    unpickles arbitrary objects, so a corrupt or untrusted demonstration file
    would execute code instead of failing.
    """
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


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
        raise TypeError(
            f"Expected list/tuple/tensor feature payload, got {type(payload)!r}"
        )

    episodes: list[torch.Tensor] = []
    for episode in payload:
        episode_tensor = (
            episode if isinstance(episode, torch.Tensor) else torch.as_tensor(episode)
        )
        episodes.extend(_tensor_to_feature_episodes(episode_tensor))
    return episodes


def _resolve_feature_order(
    feature_group: Any, expected_feature_names: list[str] | None, *, demo_key: str
) -> list[str]:
    """Decide the column order, and refuse a file that lacks a wanted feature.

    With no expectation, the HDF5 group order wins: the recorder writes with
    ``track_order=True``, so it is the reward-manager order the features were
    captured in.

    A named request is a *subset* request. One recording serves several arms,
    and the arm that holds one reward coefficient fixed learns over fewer
    columns than the recording stores; re-recording per subset would turn one
    dataset into several. A requested name the file does not hold is still an
    error - nothing can fill it in.
    """
    if not hasattr(feature_group, "keys"):
        raise ValueError(
            f"Demo '{demo_key}' uses a legacy feature format; expected a group "
            "'features/<feature_name>'. Re-record it with scripts/record_demos.py."
        )
    available = [str(name) for name in feature_group.keys()]
    if not available:
        raise ValueError(f"Demo '{demo_key}' has an empty 'features' group.")
    if expected_feature_names is None:
        return available

    expected = [str(name) for name in expected_feature_names]
    missing = [name for name in expected if name not in available]
    if missing:
        raise ValueError(
            f"Demo '{demo_key}' is missing reward features {sorted(missing)}; it "
            f"holds {sorted(available)}. Re-record the demos with this feature "
            "map, or match the ignored/forced terms it was recorded with."
        )
    return expected


def _feature_column(
    feature_node: Any, *, demo_key: str, feature_name: str
) -> torch.Tensor:
    """Read one feature's sequence as ``[T]``."""
    if hasattr(feature_node, "keys"):
        raise ValueError(
            f"Demo '{demo_key}' feature '{feature_name}' must be a dataset of "
            "shape [T] or [T,1], not a group."
        )
    values = torch.as_tensor(feature_node[...])
    if values.ndim == 2 and int(values.shape[1]) == 1:
        return rearrange(values, "t 1 -> t")
    if values.ndim != 1:
        raise ValueError(
            f"Demo '{demo_key}' feature '{feature_name}' must have shape [T] or "
            f"[T,1], got {tuple(values.shape)}."
        )
    return values


def _episode_matrix(
    feature_group: Any, feature_names: list[str], *, demo_key: str
) -> torch.Tensor:
    """Stack one demo's features into ``[T, D]`` in the given order."""
    columns: list[torch.Tensor] = []
    episode_length: int | None = None
    for feature_name in feature_names:
        values = _feature_column(
            feature_group[feature_name], demo_key=demo_key, feature_name=feature_name
        )
        num_steps = int(values.shape[0])
        if episode_length is None:
            if num_steps == 0:
                raise ValueError(
                    f"Demo '{demo_key}' has an empty feature sequence '{feature_name}'."
                )
            episode_length = num_steps
        elif num_steps != episode_length:
            raise ValueError(
                f"Demo '{demo_key}' feature lengths disagree: expected "
                f"{episode_length}, feature '{feature_name}' has {num_steps}."
            )
        columns.append(values)
    return torch.stack(columns, dim=1)


def _extract_feature_episodes_from_hdf5(
    path: str,
    expected_feature_names: list[str] | None = None,
) -> list[torch.Tensor]:
    """Read every demo's feature trajectory from an HDF5 file."""
    episodes: list[torch.Tensor] = []
    for demo_key, demo_group in iter_hdf5_demos(
        path, payload_hint="features/<feature_name>"
    ):
        if "features" not in demo_group:
            raise ValueError(f"Demo '{demo_key}' is missing the 'features' group.")
        feature_group = demo_group["features"]
        feature_names = _resolve_feature_order(
            feature_group, expected_feature_names, demo_key=demo_key
        )
        episodes.append(
            _episode_matrix(feature_group, feature_names, demo_key=demo_key)
        )
    if not episodes:
        raise ValueError(f"No demos with features found in {path}.")
    return episodes


def _validate_feature_episodes(
    episodes: list[torch.Tensor],
    expected_feature_dim: int,
) -> list[torch.Tensor]:
    if len(episodes) == 0:
        raise ValueError("No expert feature episodes found.")

    validated: list[torch.Tensor] = []
    for episode_idx, episode in enumerate(episodes):
        episode_tensor = (
            episode if isinstance(episode, torch.Tensor) else torch.as_tensor(episode)
        )
        if episode_tensor.ndim != 2:
            raise ValueError(
                f"Episode {episode_idx} must have shape [T, D], got {tuple(episode_tensor.shape)}"
            )
        if episode_tensor.shape[0] <= 0:
            raise ValueError(
                f"Episode {episode_idx} is empty. Expected at least one timestep."
            )
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
    if expected_feature_names is not None and len(expected_feature_names) != int(
        expected_feature_dim
    ):
        raise ValueError(
            f"Expected {expected_feature_dim} feature names, got {len(expected_feature_names)}."
        )

    ext = os.path.splitext(path)[1].lower()
    if ext in {".h5", ".hdf5"}:
        _validate_hdf5_feature_semantics(path)
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

    return _validate_feature_episodes(
        episodes, expected_feature_dim=expected_feature_dim
    )


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


def install_expert_features(
    irl_alg: "IrlAlgorithm",
    expert_data_path: str,
    *,
    max_num_trajectories: int | None = None,
    subset_strategy: str = "first",
    seed: int = 42,
) -> None:
    """Load expert feature episodes from disk into ``irl_alg``'s expert buffer.

    The expected feature dimension and names come from the algorithm's own
    spec, so a demonstration file recorded against a different reward space is
    rejected here rather than silently mislabeled.
    """
    spec = irl_alg.feature_spec
    expected_feature_names = list(spec.names) if spec.names is not None else None
    episodes = load_feature_episodes(
        os.path.abspath(expert_data_path),
        expected_feature_dim=int(spec.dim),
        expected_feature_names=expected_feature_names,
    )
    total = len(episodes)
    if max_num_trajectories is not None:
        episodes = subset_episodes(
            episodes, max_num_trajectories, subset_strategy, seed
        )
        print(
            f"[INFO] Expert subset: {len(episodes)}/{total} trajectories (strategy={subset_strategy})"
        )
    for episode in episodes:
        irl_alg.add_expert_episode(episode)
