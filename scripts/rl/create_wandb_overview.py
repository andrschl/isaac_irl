"""Create the curated Cartpole sample-efficiency W&B report."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity; defaults to $WANDB_ENTITY.",
    )
    parser.add_argument("--project", default="isaac-irl-cartpole-sample-efficiency")
    parser.add_argument("--group", default="cartpole-fixed-reward-v4")
    parser.add_argument(
        "--title", default="Cartpole RL sample efficiency — clean benchmark"
    )
    args = parser.parse_args()

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)

    entity = args.entity or os.environ.get("WANDB_ENTITY")
    if not entity:
        raise ValueError(
            "No W&B entity: pass --entity or set WANDB_ENTITY in the environment."
        )

    from utils.wandb_overview import create_cartpole_overview_report

    url = create_cartpole_overview_report(
        entity=entity,
        project=args.project,
        group=args.group,
        title=args.title,
    )
    print(f"W&B overview: {url}")


if __name__ == "__main__":
    main()
