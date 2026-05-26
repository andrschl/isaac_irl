"""Shared HDF5 walking helpers for expert-demo loaders."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def iter_hdf5_demos(path: str, *, payload_hint: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(demo_key, demo_group)`` for each demo under ``data/`` in
    sorted order.

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
