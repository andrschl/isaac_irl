"""Human-readable console sink for the runner's :class:`LogEvent` stream.

Subsumes the old ``ConsoleReporter`` and reuses its formatting bodies under
a single :meth:`log` method that dispatches on ``event.kind``.
"""

from __future__ import annotations

from .loggers import LogEvent


class ConsoleLogger:
    """Print concise per-event summaries to stdout."""

    def start(self, num_iterations: int) -> None:
        print(f"[INFO] Starting learning for {int(num_iterations)} iterations.")

    def finish(self, exit_code: int = 0) -> None:
        del exit_code

    def log(self, event: LogEvent) -> None:
        kind = event.kind
        if kind == "iteration_summary":
            self._iteration_summary(event)
        elif kind == "validation/summary":
            self._validation_summary(event)
        elif kind == "validation/features":
            self._validation_features(event)
        # Other kinds (e.g. raw scalar emissions for wandb-only metrics) are
        # silently ignored — the console sink curates what gets printed.

    @staticmethod
    def _iteration_summary(event: LogEvent) -> None:
        payload = event.payload
        policy_loss = payload.get("RL/policy_loss", float("nan"))
        reward_loss = payload.get("IRL/reward_loss", float("nan"))
        feat_diff = payload.get("IRL/feature_exp_diff_norm", float("nan"))
        bc_loss = payload.get("BC/loss")
        bc_str = f"  bc_loss={bc_loss:.4f}" if bc_loss is not None else ""
        print(
            f"[INFO] iter {event.iteration + 1}  policy_loss={policy_loss:.4f}  "
            f"reward_loss={reward_loss:.4f}  feat_diff_norm={feat_diff:.4f}{bc_str}"
        )

    @staticmethod
    def _validation_summary(event: LogEvent) -> None:
        if not event.payload:
            return
        summary = ", ".join(
            f"{key.rsplit('/', 1)[-1]}={value:.4f}" for key, value in event.payload.items()
        )
        label_part = f"  {event.label}" if event.label else ""
        print(f"[INFO] validation summary iter {event.iteration + 1}{label_part}  {summary}")

    @staticmethod
    def _validation_features(event: LogEvent) -> None:
        prefix = event.key_prefix
        feature_items = [
            f"{key.removeprefix(prefix)}={value:.4f}"
            for key, value in event.payload.items()
            if key.startswith(prefix)
        ]
        if not feature_items:
            return
        label_part = f"  {event.label}" if event.label else ""
        print(
            f"[INFO] validation features iter {event.iteration + 1}{label_part}  "
            + ", ".join(feature_items)
        )
