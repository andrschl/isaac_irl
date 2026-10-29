"""Launch isolated finite-horizon successor-PMD checkpoint evaluations."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT / "src", REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from training.checkpoint_evaluation import main  # noqa: E402
from training.pipeline import configure_torch_backends  # noqa: E402

if __name__ == "__main__":
    configure_torch_backends()
    main()
