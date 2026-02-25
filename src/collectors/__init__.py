"""Dataset collector entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .robomimic_data_collector import RobomimicDataCollector

__all__ = ["RobomimicDataCollector"]


def __getattr__(name: str) -> Any:
    if name == "RobomimicDataCollector":
        from .robomimic_data_collector import RobomimicDataCollector

        return RobomimicDataCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
