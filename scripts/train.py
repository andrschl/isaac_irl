# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a policy with IRL, RL, and optional BC regularization."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT / "src", REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from training.pipeline import configure_torch_backends, main  # noqa: E402

if __name__ == "__main__":
    configure_torch_backends()
    main()
