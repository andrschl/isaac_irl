"""Thin AppLauncher-aware CLI adapter for IRL training."""

from __future__ import annotations

import argparse
from typing import Any

from config.irl_cli import build_irl_arg_parser


def build_arg_parser(app_launcher_cls: Any) -> argparse.ArgumentParser:
    """Compose the pure training schema with Isaac AppLauncher arguments."""
    parser = build_irl_arg_parser()
    app_launcher_cls.add_app_launcher_args(parser)
    return parser
