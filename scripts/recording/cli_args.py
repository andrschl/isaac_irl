from __future__ import annotations

import sys
from pathlib import Path

try:
    from scripts.rsl_rl.cli_args import add_rsl_rl_args, parse_rsl_rl_cfg, update_rsl_rl_cfg
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.rsl_rl.cli_args import add_rsl_rl_args, parse_rsl_rl_cfg, update_rsl_rl_cfg


__all__ = ["add_rsl_rl_args", "parse_rsl_rl_cfg", "update_rsl_rl_cfg"]
