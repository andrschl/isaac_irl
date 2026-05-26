from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from utils.runtime_context import RuntimeContext


@dataclass(slots=True)
class ObsActionBufCfg:
    """Storage-side config for the (obs, action) buffer."""

    capacity: int | None = None  # None = grow to fit; int caps the number of pairs
    store_device: torch.device | str = "cpu"
    store_dtype: torch.dtype = torch.float32
    sample_with_replacement: bool = True


class ObsActionBuffer:
    """Flat (obs, action) pair store for supervised imitation losses.

    Obs is a nested ``dict[str, Tensor | dict]`` mirroring the TensorDict groups
    the actor consumes (typically just ``policy/...``). Internally the leaves
    are flattened to ``Tensor[N_total, *leaf_shape]`` so sampling is a single
    ``randint`` + indexing call.

    Unlike :class:`storage.feature_storage.FeatureTrajectoryBuffer`, this buffer
    does NOT preserve episode boundaries: BC consumes individual transitions
    and gains nothing from per-episode padding/masking.
    """

    def __init__(self, cfg: ObsActionBufCfg, ctx: RuntimeContext) -> None:
        self.cfg = cfg
        self.ctx = ctx
        self._store_device = torch.device(cfg.store_device)

        # Flat tensors are populated by load_episodes().
        # _obs is a nested dict mirroring the obs group structure; leaves are
        # tensors of shape [N_total, *leaf_shape].
        self._obs: dict[str, Any] = {}
        self._actions: torch.Tensor | None = None
        self._n_total: int = 0

    def __len__(self) -> int:
        return self._n_total

    @property
    def num_actions(self) -> int | None:
        if self._actions is None:
            return None
        return self._actions.shape[-1]

    def clear(self) -> None:
        self._obs = {}
        self._actions = None
        self._n_total = 0

    def load_episodes(
        self,
        episodes: list[tuple[dict[str, Any], torch.Tensor]],
    ) -> None:
        """Populate the buffer from a list of expert episodes.

        Shapes:
            ``episodes[i] = (obs_dict_i, actions_i)``
              - ``actions_i``: ``[T_i, A]``
              - each leaf of ``obs_dict_i``: ``[T_i, *leaf_shape]``
            All episodes must share the same nested obs structure and the
            same action dim ``A``. After loading, the flat buffer holds
            ``N_total = sum_i T_i`` ``(obs, action)`` pairs.
        """
        if len(episodes) == 0:
            raise ValueError("Cannot load an empty list of episodes.")

        # Validate and discover the obs leaf structure from the first episode.
        first_obs, first_actions = episodes[0]
        if first_actions.ndim != 2:
            raise ValueError(
                f"Expected actions of shape [T, A], got {tuple(first_actions.shape)}."
            )

        leaf_paths = _list_leaf_paths(first_obs)
        if len(leaf_paths) == 0:
            raise ValueError("First episode obs dict has no leaf tensors.")

        action_dim = first_actions.shape[-1]

        # Per-leaf flat lists of tensors to concatenate at the end.
        per_leaf_chunks: dict[tuple[str, ...], list[torch.Tensor]] = {
            path: [] for path in leaf_paths
        }
        action_chunks: list[torch.Tensor] = []
        total_steps = 0

        for ep_idx, (obs_dict, actions) in enumerate(episodes):
            ep_paths = _list_leaf_paths(obs_dict)
            if ep_paths != leaf_paths:
                raise ValueError(
                    f"Episode {ep_idx} obs leaf structure mismatch: "
                    f"expected {['/'.join(p) for p in leaf_paths]}, "
                    f"got {['/'.join(p) for p in ep_paths]}."
                )
            if actions.ndim != 2 or actions.shape[-1] != action_dim:
                raise ValueError(
                    f"Episode {ep_idx} actions shape mismatch: "
                    f"expected [T, {action_dim}], got {tuple(actions.shape)}."
                )

            t_steps = actions.shape[0]
            if t_steps <= 0:
                raise ValueError(f"Episode {ep_idx} has no steps.")

            for path in leaf_paths:
                leaf = _get_at_path(obs_dict, path)
                leaf_tensor = (
                    leaf if isinstance(leaf, torch.Tensor) else torch.as_tensor(leaf)
                )
                if leaf_tensor.shape[0] != t_steps:
                    raise ValueError(
                        f"Episode {ep_idx} leaf {'/'.join(path)} has shape "
                        f"{tuple(leaf_tensor.shape)}; first dim must equal "
                        f"actions T={t_steps}."
                    )
                per_leaf_chunks[path].append(
                    leaf_tensor.detach().to(
                        device=self._store_device, dtype=self.cfg.store_dtype
                    )
                )

            action_chunks.append(
                actions.detach().to(
                    device=self._store_device, dtype=self.cfg.store_dtype
                )
            )
            total_steps += t_steps

        actions_flat = torch.cat(action_chunks, dim=0)
        obs_flat_leaves: dict[tuple[str, ...], torch.Tensor] = {
            path: torch.cat(chunks, dim=0) for path, chunks in per_leaf_chunks.items()
        }

        # Apply capacity cap (keep most recent for parity with FIFO eviction).
        if self.cfg.capacity is not None and total_steps > self.cfg.capacity:
            keep = self.cfg.capacity
            actions_flat = actions_flat[-keep:]
            obs_flat_leaves = {p: t[-keep:] for p, t in obs_flat_leaves.items()}
            total_steps = keep

        self._actions = actions_flat
        self._obs = _build_nested_from_leaves(obs_flat_leaves)
        self._n_total = total_steps

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Sample a flat minibatch on ``device``.

        Returns:
            ``(obs_mb, actions_mb)`` where
              - ``actions_mb``: ``[B, A]``
              - each leaf of ``obs_mb`` mirrors the nested obs structure
                with shape ``[B, *leaf_shape]``.
        """
        if self._actions is None or self._n_total == 0:
            raise RuntimeError(
                "ObsActionBuffer is empty. Call load_episodes(...) first."
            )
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

        device = torch.device(device)

        if self.cfg.sample_with_replacement:
            idx = torch.randint(
                0, self._n_total, (batch_size,), device=self._store_device
            )
        else:
            if batch_size > self._n_total:
                raise ValueError(
                    f"`batch_size={batch_size}` exceeds buffer size {self._n_total} "
                    "with sample_with_replacement=False."
                )
            idx = torch.randperm(self._n_total, device=self._store_device)[:batch_size]

        actions_mb = self._actions.index_select(0, idx).to(device)
        obs_mb = _index_select_nested(self._obs, idx, device)
        return obs_mb, actions_mb


# ---------------------------------------------------------------------------
# Internal helpers for nested obs dicts
# ---------------------------------------------------------------------------


def _list_leaf_paths(obs: dict[str, Any]) -> list[tuple[str, ...]]:
    """Return a sorted list of leaf paths in a nested obs dict."""
    paths: list[tuple[str, ...]] = []

    def _walk(node: Any, prefix: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key in sorted(node.keys()):
                _walk(node[key], prefix + (str(key),))
        else:
            paths.append(prefix)

    _walk(obs, ())
    return paths


def _get_at_path(obs: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = obs
    for key in path:
        node = node[key]
    return node


def _build_nested_from_leaves(
    leaves: dict[tuple[str, ...], torch.Tensor],
) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path, tensor in leaves.items():
        node = root
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = tensor
    return root


def _index_select_nested(
    obs: dict[str, Any],
    idx: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            out[key] = _index_select_nested(value, idx, device)
        else:
            out[key] = value.index_select(0, idx).to(device)
    return out
