"""Launch a bounded, two-learner PMD follow-up to a completed screening sweep."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.parallel_sweep import main  # noqa: E402

if __name__ == "__main__":
    main()
