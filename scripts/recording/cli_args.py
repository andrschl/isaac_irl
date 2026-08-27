"""Re-export the rsl_rl CLI helpers from scripts/rsl_rl/cli_args.py.

The recorder runs as a script, so Python puts ``scripts/recording/`` on
``sys.path`` and ``import cli_args`` resolves to *this* module. We must not
re-import the name ``cli_args`` here: it is already in ``sys.modules`` (this
file, mid-initialization), so ``from cli_args import ...`` would resolve back
to ourselves and raise a circular-import error.

Instead we load the sibling ``scripts/rsl_rl/cli_args.py`` directly from its
path under a distinct module name. No ``sys.path`` games, no namespace-package
discovery, no name collision.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_rsl_rl_cli_args_path = Path(__file__).resolve().parents[1] / "rsl_rl" / "cli_args.py"
_spec = importlib.util.spec_from_file_location(
    "_rsl_rl_cli_args", _rsl_rl_cli_args_path
)
if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
    raise ImportError(f"Could not load rsl_rl CLI helpers from {_rsl_rl_cli_args_path}")
_rsl_rl_cli_args = importlib.util.module_from_spec(_spec)
sys.modules["_rsl_rl_cli_args"] = _rsl_rl_cli_args
_spec.loader.exec_module(_rsl_rl_cli_args)

add_rsl_rl_args = _rsl_rl_cli_args.add_rsl_rl_args
parse_rsl_rl_cfg = _rsl_rl_cli_args.parse_rsl_rl_cfg
update_rsl_rl_cfg = _rsl_rl_cli_args.update_rsl_rl_cfg

__all__ = ["add_rsl_rl_args", "parse_rsl_rl_cfg", "update_rsl_rl_cfg"]
