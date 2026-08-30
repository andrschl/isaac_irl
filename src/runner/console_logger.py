"""Human-readable console sink for the runner's :class:`LogEvent` stream.

Subsumes the old ``ConsoleReporter`` and reuses its formatting bodies under
a single :meth:`log` method that dispatches on ``event.kind``.
"""

from __future__ import annotations

from .loggers import LogEvent


class ConsoleLogger:
    """Print concise per-event summaries to stdout."""

    def start(self, num_iterations: int) -> None:
        print(
            f"[INFO] Starting learning for {int(num_iterations)} iterations.",
            flush=True,
        )

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
        fields: list[str] = []
        for label, key in (
            ("critic_loss", "Overview/critic_loss"),
            ("actor_delta", "Overview/actor_parameter_delta"),
            ("policy_std", "Overview/policy_std"),
            ("q_mean", "Overview/q_mean"),
            ("train_reward", "Overview/train_reward_per_step"),
            ("act_sat", "Overview/action_saturation_fraction"),
            ("psi_scale", "Overview/successor_scale_max"),
            ("psi_refits", "Overview/successor_scale_refits"),
            ("alpha", "Overview/entropy_temperature"),
            ("actor_kl", "Overview/pmd_actor_kl"),
            ("policy_updates", "Overview/policy_updates"),
            ("timeout", "Overview/timeout_fraction"),
            ("no_bootstrap", "Overview/bootstrap_disabled_fraction"),
            ("clip", "Overview/action_clipped_fraction"),
            ("reward_error", "Overview/reward_reconstruction_error"),
            ("policy_loss", "RL/policy_loss"),
            ("reward_loss", "IRL/reward_loss"),
            ("feat_diff_norm", "IRL/feature_exp_diff_norm"),
            ("bc_loss", "BC/loss"),
        ):
            if key in payload:
                fields.append(f"{label}={payload[key]:.4g}")
        summary = "  ".join(fields) if fields else "no summary metrics"
        print(f"[INFO] iter {event.iteration + 1}  {summary}", flush=True)

    @staticmethod
    def _validation_summary(event: LogEvent) -> None:
        if not event.payload:
            return
        summary = ", ".join(
            f"{key.rsplit('/', 1)[-1]}={value:.4f}"
            for key, value in event.payload.items()
        )
        label_part = f"  {event.label}" if event.label else ""
        print(
            f"[INFO] validation summary iter {event.iteration + 1}{label_part}  {summary}",
            flush=True,
        )

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
            + ", ".join(feature_items),
            flush=True,
        )
