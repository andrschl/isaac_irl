"""Expert demo loaders for the train_irl script.

Reads expert feature episodes (``.pt``/``.pth``/``.h5``/``.hdf5``) and expert
(obs, action) episodes (``.h5``/``.hdf5`` written by
``collectors.RobomimicDataCollector``). Also provides the trajectory subsetting
helpers and ``_install_expert_episodes`` — the one-shot setup that loads expert
feature episodes from disk into a ``FeatureRewardLearner``.

Module-level imports are kept light so this file can be loaded in tests
without triggering the Isaac Lab dependency tree.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeVar

import torch
from einops import rearrange

# Ensure sibling imports and src/ imports resolve under both invocation
# patterns (`python scripts/irl/train_irl.py` and importlib loading in tests).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from algorithms import FeatureRewardLearner


__all__ = [
    "load_feature_episodes",
    "load_obs_action_episodes",
    "load_expert_success_rate",
    "_install_expert_episodes",
    "_subset_episodes",
    "_split_holdout_episodes",
]


_EpisodeT = TypeVar("_EpisodeT")


# ---------------------------------------------------------------------------
# Feature episodes (.pt/.pth or .h5/.hdf5)
# ---------------------------------------------------------------------------


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


def _iter_hdf5_demos(path: str, *, payload_hint: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(demo_key, demo_group)`` for each demo under ``data/`` in sorted order.

    Raises a clear error if ``data/`` is missing or any demo entry isn't a
    group. ``payload_hint`` is folded into the error message so the caller
    can describe what each demo should contain (e.g. ``"features/<name>"``).
    """
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Loading .h5/.hdf5 expert data requires `h5py`. Install it in your environment."
        ) from exc

    with h5py.File(path, "r") as file_handle:
        data_group = file_handle.get("data", None)
        if data_group is None or not hasattr(data_group, "keys"):
            raise ValueError(
                f"HDF5 payload must contain demos under 'data/demo_*/{payload_hint}'."
            )
        for demo_key in sorted(data_group.keys()):
            demo_group = data_group[demo_key]
            if not hasattr(demo_group, "keys"):
                raise ValueError(f"Demo '{demo_key}' must be a group.")
            yield str(demo_key), demo_group


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
    for demo_key, demo_group in _iter_hdf5_demos(path, payload_hint="features/<feature_name>"):
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
    for _demo_key, demo_group in _iter_hdf5_demos(path, payload_hint="success"):
        if "success" not in demo_group:
            continue
        values = torch.as_tensor(demo_group["success"][...]).reshape(-1)
        if values.numel() == 0:
            continue
        per_demo_success.append(float((values > 0.5).any().item()))

    if not per_demo_success:
        return None
    return sum(per_demo_success) / len(per_demo_success)


# ---------------------------------------------------------------------------
# (obs, action) episodes (.h5/.hdf5 only) — used by BC
# ---------------------------------------------------------------------------


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

    episodes: list[tuple[dict[str, Any], torch.Tensor]] = []
    for demo_key, demo_group in _iter_hdf5_demos(path, payload_hint="{obs,actions}"):
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


# ---------------------------------------------------------------------------
# Subsetting / hold-out split / installation
# ---------------------------------------------------------------------------


def _subset_episodes(
    episodes: list[_EpisodeT],
    max_num: int,
    strategy: str,
    seed: int,
) -> list[_EpisodeT]:
    """Return up to ``max_num`` episodes. ``strategy``: 'first' | 'random'."""
    if len(episodes) <= max_num:
        return episodes
    if strategy == "first":
        return episodes[:max_num]
    if strategy == "random":
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(episodes), generator=rng)[:max_num].tolist()
        return [episodes[i] for i in indices]
    raise ValueError(
        f"Unknown expert_subset_strategy '{strategy}'. Expected 'first' or 'random'."
    )


def _split_holdout_episodes(
    episodes: list[_EpisodeT],
    val_fraction: float,
    seed: int,
) -> tuple[list[_EpisodeT], list[_EpisodeT]]:
    """Split episodes into ``(train, val)`` by a held-out *episode* fraction.

    The split is at the episode level (not per-transition) so no trajectory
    leaks across train/val. ``val_fraction`` is rounded down but always leaves
    at least one episode in each side when the split is requested and possible.
    """
    n = len(episodes)
    if val_fraction <= 0.0 or n < 2:
        return episodes, []
    n_val = max(1, min(n - 1, int(n * val_fraction)))
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=rng).tolist()
    val_idx = set(perm[:n_val])
    val = [episodes[i] for i in range(n) if i in val_idx]
    train = [episodes[i] for i in range(n) if i not in val_idx]
    return train, val


def _install_expert_episodes(
    irl_alg: FeatureRewardLearner,
    expert_data_path: str,
    *,
    expected_feature_dim: int,
    expected_feature_names: list[str] | None = None,
    max_num_trajectories: int | None = None,
    subset_strategy: str = "first",
    seed: int = 42,
) -> None:
    """Load expert feature episodes from disk and add them to ``irl_alg``.

    The reward learner's expert storage must already be initialized with
    ``num_envs == 1`` (the runner does this during construction).
    """
    if irl_alg.expert_storage is None:
        raise RuntimeError("Reward learner expert storage is not initialized.")
    expert_ctx = getattr(irl_alg.expert_storage, "ctx", None)
    if expert_ctx is None or int(expert_ctx.num_envs) != 1:
        raise ValueError(
            "Expert loader requires reward learner expert storage with num_envs == 1, "
            f"got {None if expert_ctx is None else int(expert_ctx.num_envs)}."
        )

    episodes = load_feature_episodes(
        os.path.abspath(expert_data_path),
        expected_feature_dim=expected_feature_dim,
        expected_feature_names=expected_feature_names,
    )
    total = len(episodes)
    if max_num_trajectories is not None:
        episodes = _subset_episodes(
            episodes, max_num_trajectories, subset_strategy, seed
        )
        print(
            f"[INFO] Expert subset: {len(episodes)}/{total} trajectories (strategy={subset_strategy})"
        )
    for episode in episodes:
        irl_alg.add_expert_episode(episode)
