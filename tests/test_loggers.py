"""Tests for the runner's metric loggers."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


def _install_fake_wandb(
    monkeypatch, *, init_should_fail: bool = False
) -> dict[str, Any]:
    """Install a minimal fake wandb module and return the call-record dict."""
    record: dict[str, Any] = {"init_calls": 0, "finish_calls": 0, "log_calls": []}

    class _FakeRun:
        def log(self, payload, step):
            record["log_calls"].append((step, dict(payload)))

    def _init(**kwargs):
        record["init_calls"] += 1
        record["last_init_kwargs"] = kwargs
        if init_should_fail:
            raise RuntimeError("boom")
        return _FakeRun()

    def _finish(exit_code=0):
        record["finish_calls"] += 1
        record["last_finish_exit_code"] = exit_code

    class _Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = _init
    fake_wandb.finish = _finish
    fake_wandb.Settings = _Settings

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    return record


def test_wandb_logger_start_is_idempotent(monkeypatch):
    record = _install_fake_wandb(monkeypatch)
    from runner.loggers import WandbMetricLogger

    logger = WandbMetricLogger(project="test", log_dir=None)
    logger.start()
    logger.start()
    logger.start()

    assert record["init_calls"] == 1


def test_wandb_logger_log_starts_lazily(monkeypatch):
    record = _install_fake_wandb(monkeypatch)
    from runner.loggers import WandbMetricLogger

    logger = WandbMetricLogger(project="test", log_dir=None)
    logger.log({"a": 1.0}, step=10)

    assert record["init_calls"] == 1
    assert record["log_calls"] == [(10, {"a": 1.0})]


def test_wandb_logger_log_skips_empty_payload(monkeypatch):
    record = _install_fake_wandb(monkeypatch)
    from runner.loggers import WandbMetricLogger

    logger = WandbMetricLogger(project="test", log_dir=None)
    logger.log({}, step=0)

    assert record["init_calls"] == 0
    assert record["log_calls"] == []


def test_wandb_logger_finish_is_safe_when_init_failed(monkeypatch):
    record = _install_fake_wandb(monkeypatch, init_should_fail=True)
    from runner.loggers import WandbMetricLogger

    logger = WandbMetricLogger(project="test", log_dir=None)
    logger.start()
    logger.finish(exit_code=0)

    # init failed -> _run is None -> finish must short-circuit, NOT call wandb.finish.
    assert record["finish_calls"] == 0


def test_wandb_logger_finish_passes_exit_code(monkeypatch):
    record = _install_fake_wandb(monkeypatch)
    from runner.loggers import WandbMetricLogger

    logger = WandbMetricLogger(project="test", log_dir=None)
    logger.start()
    logger.finish(exit_code=7)

    assert record["finish_calls"] == 1
    assert record["last_finish_exit_code"] == 7


def test_noop_logger_does_not_explode():
    from runner.loggers import NoopMetricLogger

    logger = NoopMetricLogger()
    logger.start()
    logger.log({"a": 1.0}, step=5)
    logger.finish(exit_code=0)
