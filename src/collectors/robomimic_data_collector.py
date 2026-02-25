from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class _CollectorState:
    num_envs: int | None = None
    num_written_episodes: int = 0
    num_since_last_flush: int = 0
    total_samples: int = 0


class RobomimicDataCollector:
    """
    Minimal Robomimic-style collector writing HDF5 episode groups:
      data/demo_<id>/{obs, actions, rewards, dones, ...}

    Mapping-like payloads are flattened into slash-separated leaf keys, e.g.:
      obs/policy/joint_pos
      features/reaching_object
    """

    def __init__(
        self,
        task_name: str,
        output_dir: str,
        filename: str,
        num_demos: int,
        flush_freq: int = 1,
    ) -> None:
        self.task_name = str(task_name)
        self.output_dir = os.path.abspath(output_dir)
        self.filename = str(filename)
        self.file_path = os.path.join(self.output_dir, self.filename)
        self.num_demos = int(num_demos)
        self.flush_freq = int(max(1, flush_freq))

        self._state = _CollectorState()

        self._h5_file: Any | None = None
        self._h5_data_group: Any | None = None

        # Per-env episode buffers:
        #   env_buffer[leaf_key] -> list of per-step tensors [*]
        self._episode_buffers: list[defaultdict[str, list[torch.Tensor]]] = []

    def reset(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        try:
            import h5py
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Recording demos requires `h5py`. Install it in your environment."
            ) from exc
        self._h5_file = h5py.File(self.file_path, "w", track_order=True)
        self._h5_data_group = self._h5_file.create_group("data", track_order=True)
        self._h5_data_group.attrs["total"] = 0
        self._h5_data_group.attrs["env_name"] = self.task_name
        self._state = _CollectorState()
        self._episode_buffers = []

        print(
            f"Dataset collector {type(self).__name__} storing trajectories in: {self.output_dir}\n"
            f"    Number of demos for collection   : {self.num_demos}\n"
            f"    Frequency for saving data to disk: {self.flush_freq}\n"
        )

    def is_stopped(self) -> bool:
        return self._state.num_written_episodes >= self.num_demos

    def add(self, key: str, value: Any) -> None:
        if self.is_stopped():
            return

        leaf_tensors = self._flatten_to_leaf_tensors(root_key=str(key), value=value)
        if len(leaf_tensors) == 0:
            return

        if self._state.num_envs is None:
            first_leaf = next(iter(leaf_tensors.values()))
            if first_leaf.ndim == 0:
                raise ValueError(f"Expected batched tensor for key '{key}', got scalar tensor.")
            self._state.num_envs = int(first_leaf.shape[0])
            self._episode_buffers = [defaultdict(list) for _ in range(self._state.num_envs)]

        num_envs = int(self._state.num_envs)

        for leaf_key, leaf_tensor in leaf_tensors.items():
            if leaf_tensor.ndim == 0 or int(leaf_tensor.shape[0]) != num_envs:
                raise ValueError(
                    f"Batch mismatch for key '{leaf_key}': expected first dim {num_envs}, "
                    f"got shape {tuple(leaf_tensor.shape)}"
                )

            # Copy the full batch once, then slice per env on CPU.
            leaf_cpu = leaf_tensor.detach().cpu()
            for env_id in range(num_envs):
                step_value = leaf_cpu[env_id]
                env_steps = self._episode_buffers[env_id][leaf_key]
                if env_steps and tuple(env_steps[-1].shape) != tuple(step_value.shape):
                    raise ValueError(
                        f"Shape changed within episode for key '{leaf_key}': "
                        f"previous {tuple(env_steps[-1].shape)} vs current {tuple(step_value.shape)}"
                    )
                env_steps.append(step_value)

    def flush(self, env_ids: torch.Tensor | list[int] | tuple[int, ...]) -> None:
        if self._h5_file is None or self._h5_data_group is None:
            raise RuntimeError("Collector is not initialized. Call reset() first.")
        if self._state.num_envs is None:
            return

        env_id_list = self._normalize_env_ids(env_ids)
        if len(env_id_list) == 0:
            return

        for env_id in env_id_list:
            if self.is_stopped():
                break

            env_index = int(env_id)
            if env_index < 0 or env_index >= self._state.num_envs:
                continue

            env_buffer = self._episode_buffers[env_index]
            if len(env_buffer) == 0:
                continue

            episode_len = self._episode_length(env_buffer)
            if episode_len <= 0:
                self._episode_buffers[env_index] = defaultdict(list)
                continue

            demo_name = f"demo_{self._state.num_written_episodes}"
            demo_group = self._h5_data_group.create_group(demo_name, track_order=True)
            demo_group.attrs["num_samples"] = int(episode_len)

            for leaf_key, step_values in env_buffer.items():
                if len(step_values) == 0:
                    continue
                if len(step_values) != episode_len:
                    raise ValueError(
                        f"Episode key '{leaf_key}' has inconsistent length {len(step_values)}; "
                        f"expected {episode_len}."
                    )
                stacked_values = torch.stack(step_values, dim=0)
                self._write_dataset_by_path(demo_group, leaf_key, stacked_values)

            self._state.num_written_episodes += 1
            self._state.num_since_last_flush += 1
            self._state.total_samples += int(episode_len)
            self._h5_data_group.attrs["total"] = int(self._state.total_samples)

            self._episode_buffers[env_index] = defaultdict(list)

            if self._state.num_since_last_flush >= self.flush_freq:
                self._h5_file.flush()
                self._state.num_since_last_flush = 0

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.flush()
            self._h5_file.close()
            self._h5_file = None
            self._h5_data_group = None

    def _flatten_to_leaf_tensors(self, root_key: str, value: Any) -> dict[str, torch.Tensor]:
        if self._is_mapping_like(value):
            leaf_tensors: dict[str, torch.Tensor] = {}
            for sub_key in value.keys():
                sub_path = self._join_key_path(root_key, self._key_to_str(sub_key))
                nested_value = value[sub_key]
                nested_leaf_tensors = self._flatten_to_leaf_tensors(sub_path, nested_value)
                for leaf_path, leaf_tensor in nested_leaf_tensors.items():
                    if leaf_path in leaf_tensors:
                        raise ValueError(f"Duplicate tensor leaf path '{leaf_path}' while flattening '{root_key}'.")
                    leaf_tensors[leaf_path] = leaf_tensor
            return leaf_tensors

        tensor_value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return {root_key: tensor_value}

    @staticmethod
    def _key_to_str(key: Any) -> str:
        if isinstance(key, tuple):
            return "/".join(str(part) for part in key)
        return str(key)

    @staticmethod
    def _join_key_path(prefix: str, suffix: str) -> str:
        clean_prefix = prefix.strip("/")
        clean_suffix = suffix.strip("/")
        if not clean_prefix:
            return clean_suffix
        if not clean_suffix:
            return clean_prefix
        return f"{clean_prefix}/{clean_suffix}"

    @staticmethod
    def _is_mapping_like(value: Any) -> bool:
        return hasattr(value, "keys") and hasattr(value, "__getitem__")

    @staticmethod
    def _normalize_env_ids(env_ids: torch.Tensor | list[int] | tuple[int, ...]) -> list[int]:
        if isinstance(env_ids, torch.Tensor):
            raw_ids = env_ids.detach().reshape(-1).tolist()
        else:
            raw_ids = list(env_ids)

        # Deduplicate while preserving order.
        seen: set[int] = set()
        normalized: list[int] = []
        for env_id in raw_ids:
            env_index = int(env_id)
            if env_index in seen:
                continue
            seen.add(env_index)
            normalized.append(env_index)
        return normalized

    @staticmethod
    def _episode_length(env_buffer: dict[str, list[torch.Tensor]]) -> int:
        first_steps = next(iter(env_buffer.values()), None)
        if first_steps is None:
            return 0
        return int(len(first_steps))

    @staticmethod
    def _write_dataset_by_path(root_group: Any, key_path: str, values: torch.Tensor) -> None:
        path_parts = [part for part in str(key_path).split("/") if part]
        if len(path_parts) == 0:
            raise ValueError(f"Invalid empty dataset key path: '{key_path}'.")

        group = root_group
        for part in path_parts[:-1]:
            if part in group:
                next_group = group[part]
                if not hasattr(next_group, "create_group"):
                    raise ValueError(f"Dataset path collision at '{part}' while writing '{key_path}'.")
                group = next_group
            else:
                group = group.create_group(part, track_order=True)

        dataset_name = path_parts[-1]
        if dataset_name in group:
            raise ValueError(f"Dataset path collision for key '{key_path}'.")

        group.create_dataset(dataset_name, data=values.numpy())
