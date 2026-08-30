"""Tests for the runner's logger event sinks."""

from __future__ import annotations

import sys
import types
from typing import Any


def _install_fake_wandb(
    monkeypatch, *, init_should_fail: bool = False
) -> dict[str, Any]:
    """Install a minimal fake wandb module and return the call-record dict."""
    record: dict[str, Any] = {
        "init_calls": 0,
        "finish_calls": 0,
        "define_metric_calls": [],
        "log_calls": [],
        "video_calls": [],
    }

    class _FakeRun:
        url = "https://wandb.test/run"

        def define_metric(self, name, **kwargs):
            record["define_metric_calls"].append((name, kwargs))

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

    def _video(path, format):
        record["video_calls"].append((path, format))
        return {"path": path, "format": format}

    class _Settings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.init = _init
    fake_wandb.finish = _finish
    fake_wandb.Video = _video
    fake_wandb.Settings = _Settings

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    return record


def _event(payload, step=0, **kwargs):
    from runner.loggers import LogEvent

    return LogEvent(kind="iteration_summary", step=step, payload=payload, **kwargs)


def test_console_logger_reports_overview_metrics_without_placeholder_nan(capsys):
    from runner.console_logger import ConsoleLogger

    logger = ConsoleLogger()
    logger.log(
        _event(
            {
                "Overview/critic_loss": 0.25,
                "Overview/actor_parameter_delta": 0.1,
                "Overview/reward_reconstruction_error": 0.0,
            },
            step=4_000,
        )
    )
    output = capsys.readouterr().out
    assert "critic_loss=0.25" in output
    assert "actor_delta=0.1" in output
    assert "reward_error=0" in output
    assert "nan" not in output.lower()


def test_wandb_logger_start_is_idempotent(monkeypatch):
    record = _install_fake_wandb(monkeypatch)
    from runner.loggers import WandbLogger

    logger = WandbLogger(project="test", log_dir=None)
    logger.start(num_iterations=10)
    logger.start(num_iterations=10)
    logger.start(num_iterations=10)

    assert record["init_calls"] == 1
    assert record["define_metric_calls"] == [
        ("Data/simulator_transitions_total", {"summary": "max"}),
        (
            "Overview/*",
            {
                "step_metric": "Data/simulator_transitions_total",
                "summary": "last",
            },
        ),
    ]


def test_wandb_logger_log_starts_lazily(monkeypatch):
    record = _install_fake_wandb(monkeypatch)
    from runner.loggers import WandbLogger

    logger = WandbLogger(project="test", log_dir=None)
    logger.log(_event({"a": 1.0}, step=10))

    assert record["init_calls"] == 1
    assert record["log_calls"] == [(10, {"a": 1.0})]


def test_wandb_logger_log_skips_empty_payload(monkeypatch):
    record = _install_fake_wandb(monkeypatch)
    from runner.loggers import WandbLogger

    logger = WandbLogger(project="test", log_dir=None)
    logger.log(_event({}, step=0))

    assert record["init_calls"] == 0
    assert record["log_calls"] == []


def test_wandb_logger_finish_is_safe_when_init_failed(monkeypatch):
    record = _install_fake_wandb(monkeypatch, init_should_fail=True)
    from runner.loggers import WandbLogger

    logger = WandbLogger(project="test", log_dir=None)
    logger.start(num_iterations=1)
    logger.finish(exit_code=0)

    # init failed -> _run is None -> finish must short-circuit, NOT call wandb.finish.
    assert record["finish_calls"] == 0


def test_wandb_logger_finish_passes_exit_code(monkeypatch):
    record = _install_fake_wandb(monkeypatch)
    from runner.loggers import WandbLogger

    logger = WandbLogger(project="test", log_dir=None)
    logger.start(num_iterations=1)
    logger.finish(exit_code=7)

    assert record["finish_calls"] == 1
    assert record["last_finish_exit_code"] == 7


def test_wandb_logger_uploads_promoted_final_videos(monkeypatch, tmp_path):
    record = _install_fake_wandb(monkeypatch)
    from runner.loggers import WandbLogger

    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    inference_path = videos_dir / "final_inference.mp4"
    stochastic_path = videos_dir / "final_stochastic.mp4"
    inference_path.write_bytes(b"fake-mp4")
    stochastic_path.write_bytes(b"fake-mp4")

    logger = WandbLogger(project="test", log_dir=str(tmp_path))
    logger.log(_event({"a": 1.0}, step=12))
    logger.finish(exit_code=0)

    assert record["video_calls"] == [
        (str(inference_path), "mp4"),
        (str(stochastic_path), "mp4"),
    ]
    video_step, payload = record["log_calls"][-1]
    assert video_step == 12
    assert set(payload) == {
        "Videos/final_inference",
        "Videos/final_stochastic",
    }


def test_noop_logger_does_not_explode():
    from runner.loggers import NoopLogger

    logger = NoopLogger()
    logger.start(num_iterations=5)
    logger.log(_event({"a": 1.0}, step=5))
    logger.finish(exit_code=0)


def test_composite_logger_fans_out():
    from runner.loggers import CompositeLogger, LogEvent

    class _Recorder:
        def __init__(self):
            self.starts: list[int] = []
            self.events: list[LogEvent] = []
            self.finishes: list[int] = []

        def start(self, num_iterations):
            self.starts.append(num_iterations)

        def log(self, event):
            self.events.append(event)

        def finish(self, exit_code=0):
            self.finishes.append(exit_code)

    a, b = _Recorder(), _Recorder()
    logger = CompositeLogger([a, b])
    logger.start(7)
    evt = LogEvent(kind="iteration_summary", step=1, payload={"x": 1.0})
    logger.log(evt)
    logger.finish(exit_code=3)

    for r in (a, b):
        assert r.starts == [7]
        assert r.events == [evt]
        assert r.finishes == [3]


def test_console_keeps_nested_validation_keys_distinguishable(capsys) -> None:
    """Both policy modes report the same leaf name; the mode must survive."""
    from runner.console_logger import ConsoleLogger
    from runner.loggers import LogEvent

    ConsoleLogger().log(
        LogEvent(
            kind="validation/summary",
            step=1024,
            payload={
                "Validation/environment_return_mean": 0.70,
                "Validation/stochastic/environment_return_mean": 0.72,
            },
        )
    )

    printed = capsys.readouterr().out
    assert "environment_return_mean=0.7000" in printed
    assert "stochastic/environment_return_mean=0.7200" in printed
