"""Launch offline std calibration from complete MSE BC checkpoints."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_path in (REPO_ROOT / "src", REPO_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from training.spread_calibration import main  # noqa: E402

if __name__ == "__main__":
    main()
