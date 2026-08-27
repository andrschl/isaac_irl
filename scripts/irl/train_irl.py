# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin launcher for IRL and fixed-reward robotics training."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_SRC = REPO_ROOT / "src"
for search_path in (REPO_SRC, REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from adapters.rsl_rl_isaac_training import main as _main  # noqa: E402


def main(argv: list[str] | None = None, deps: Any | None = None) -> None:
    """Parse configuration, construct dependencies, and run training."""
    _main(argv=argv, deps=deps)


if __name__ == "__main__":
    main()
