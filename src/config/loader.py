"""Build configuration dataclasses from plain mappings, strictly.

One recursive builder replaces a per-field ``section.get(name, default)``
chain. Unknown keys are an error rather than a silent no-op, which is what
catches a renamed or misspelled setting at load time instead of at the end of a
run that quietly used the default.
"""

from __future__ import annotations

import types
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import torch

T = TypeVar("T")

_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def build_dataclass(cls: type[T], mapping: Any, *, path: str = "") -> T:
    """Construct ``cls`` from ``mapping``, rejecting keys it does not declare.

    ``path`` names the config section in error messages ("runner.expert_buffer"),
    so a bad key reports where it was written, not just what it was.
    """
    section = _as_mapping(mapping, path=path)
    hints = get_type_hints(cls)
    declared = {field.name for field in fields(cls)}
    unknown = sorted(set(section) - declared)
    if unknown:
        where = path or cls.__name__
        raise ValueError(
            f"Unknown key(s) {unknown} in `{where}`. "
            f"Valid keys are: {sorted(declared)}."
        )

    values: dict[str, Any] = {}
    for field in fields(cls):
        if field.name not in section:
            continue
        field_path = f"{path}.{field.name}" if path else field.name
        values[field.name] = _coerce(
            section[field.name], hints[field.name], path=field_path
        )
    return cls(**values)


def _as_mapping(payload: Any, *, path: str) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        where = path or "config"
        raise TypeError(f"Config section `{where}` must be a mapping, got {payload!r}.")
    return payload


def _coerce(value: Any, annotation: Any, *, path: str) -> Any:
    # An already-built config (a discriminated union the caller resolved) is
    # taken as given; the loader only builds what arrived as a mapping.
    if is_dataclass(value) and not isinstance(value, type):
        return value

    origin = get_origin(annotation)

    if origin in (Union, types.UnionType):
        options = [arg for arg in get_args(annotation) if arg is not type(None)]
        if value is None:
            return None
        if len(options) != 1:
            raise TypeError(
                f"`{path}` has an ambiguous type {annotation}; give it an explicit "
                "parser instead of relying on the generic loader."
            )
        return _coerce(value, options[0], path=path)

    if is_dataclass(annotation):
        return build_dataclass(annotation, value, path=path)

    if origin is tuple:
        item_type = get_args(annotation)[0]
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"`{path}` must be a sequence, got {value!r}.")
        return tuple(_coerce(item, item_type, path=path) for item in value)

    if annotation is torch.dtype:
        return _torch_dtype(value, path=path)

    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"`{path}` must be true or false, got {value!r}.")
        return value

    if annotation in (int, float, str):
        return annotation(value)

    return value


def _torch_dtype(value: Any, *, path: str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value not in _TORCH_DTYPES:
        raise ValueError(
            f"`{path}` must be one of {sorted(_TORCH_DTYPES)}, got {value!r}."
        )
    return _TORCH_DTYPES[value]


def field_default(cls: type, name: str) -> Any:
    """Return the declared default of one dataclass field."""
    for field in fields(cls):
        if field.name == name:
            if field.default is not MISSING:
                return field.default
            if field.default_factory is not MISSING:  # type: ignore[misc]
                return field.default_factory()  # type: ignore[misc]
            raise ValueError(f"`{cls.__name__}.{name}` has no default.")
    raise AttributeError(f"{cls.__name__} has no field {name!r}.")
