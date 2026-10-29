"""Launch a sequential, time-budgeted successor-PMD tuning experiment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.sequential_sweep import main  # noqa: E402

if __name__ == "__main__":
    main()
