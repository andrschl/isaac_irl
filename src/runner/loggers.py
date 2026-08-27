"""Logger event stream consumed by the runner.

The runner emits one :class:`LogEvent` per logical training event (iteration
summary, validation summary, etc.) into a single :class:`Logger`. Sinks decide
what to do with each event — :class:`WandbLogger` forwards the payload to
wandb, :class:`ConsoleLogger` formats human-readable lines, :class:`NoopLogger`
drops everything. :class:`CompositeLogger` fans out to a list.

This replaces the older split between ``MetricLogger`` (wandb) and
``ConsoleReporter`` (stdout) the runner used to drive separately.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


@dataclass(slots=True)
class LogEvent:
    """One logical training event.

    ``payload`` carries the scalar metrics (wandb key → float). Optional
    ``label`` and ``key_prefix`` are read by the console sink to format the
    line; the wandb sink ignores them. ``iteration`` is used by the console
    sink for per-iter headers; the wandb sink reads ``step`` only.
    """

    kind: str
    step: int
    iteration: int = 0
    payload: dict[str, float] = field(default_factory=dict)
    label: str = ""
    key_prefix: str = ""


class Logger(Protocol):
    def start(self, num_iterations: int) -> None: ...
    def log(self, event: LogEvent) -> None: ...
    def finish(self, exit_code: int = 0) -> None: ...


class NoopLogger:
    """Logger that drops everything. Useful for tests and ``--logger noop``."""

    def start(self, num_iterations: int) -> None:
        del num_iterations

    def log(self, event: LogEvent) -> None:
        del event

    def finish(self, exit_code: int = 0) -> None:
        del exit_code


class CompositeLogger:
    """Fan-out logger: forwards each call to every wrapped logger."""

    def __init__(self, loggers: Iterable[Logger]) -> None:
        self._loggers = list(loggers)

    def start(self, num_iterations: int) -> None:
        for logger in self._loggers:
            logger.start(num_iterations)

    def log(self, event: LogEvent) -> None:
        for logger in self._loggers:
            logger.log(event)

    def finish(self, exit_code: int = 0) -> None:
        for logger in self._loggers:
            logger.finish(exit_code)


class WandbLogger:
    """Forward ``event.payload`` to Weights & Biases at ``event.step``.

    ``finish`` must be called explicitly because Isaac's ``SimulationApp.close``
    exits via ``os._exit`` and can skip Python atexit handlers.
    """

    def __init__(
        self,
        *,
        project: str,
        log_dir: str | None,
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.project = str(project)
        self.log_dir = log_dir
        self.run_name = run_name
        self.group = group
        self.tags = list(tags) if tags else None
        self.config = dict(config) if config else None
        self._run: Any = None
        self._last_step = 0

    def start(self, num_iterations: int) -> None:
        del num_iterations
        if self._run is not None:
            return
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
        try:
            import wandb

            self._run = wandb.init(
                project=self.project,
                dir=self.log_dir,
                name=self.run_name,
                group=self.group,
                tags=self.tags,
                config=self.config,
                reinit="finish_previous",
                settings=wandb.Settings(console="off"),
            )
            self._run.define_metric("Data/simulator_transitions_total", summary="max")
            self._run.define_metric(
                "Overview/*",
                step_metric="Data/simulator_transitions_total",
                summary="last",
            )
            run_url = getattr(self._run, "url", None)
            if run_url:
                print(f"[INFO] W&B run: {run_url}", flush=True)
        except Exception as exc:
            print(f"[WARN] wandb.init failed: {exc}")
            self._run = None

    def log(self, event: LogEvent) -> None:
        if not event.payload:
            return
        if self._run is None:
            self.start(num_iterations=0)
        if self._run is not None:
            self._run.log(event.payload, step=event.step)
            self._last_step = int(event.step)

    def _log_final_videos(self, wandb_module: Any) -> None:
        """Upload promoted final videos from ``log_dir/videos`` when present."""

        if self._run is None or self.log_dir is None:
            return
        videos_dir = os.path.join(self.log_dir, "videos")
        video_names = (
            "final_inference.mp4",
            "final_stochastic.mp4",
        )
        video_payload = {
            f"Videos/{os.path.splitext(file_name)[0]}": wandb_module.Video(
                os.path.join(videos_dir, file_name), format="mp4"
            )
            for file_name in video_names
            if os.path.isfile(os.path.join(videos_dir, file_name))
        }
        if video_payload:
            self._run.log(video_payload, step=self._last_step)

    def finish(self, exit_code: int = 0) -> None:
        if self._run is None:
            return
        try:
            import wandb

            self._log_final_videos(wandb)
            wandb.finish(exit_code=exit_code)
        except Exception as exc:
            print(f"[WARN] wandb.finish failed: {exc}")
        finally:
            self._run = None
