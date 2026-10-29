"""Human-readable console sink for the runner's :class:`LogEvent` stream.

The console prints exactly the ``Summary/*`` contract, so what scrolls past in
a terminal and what appears on the run dashboard are the same numbers.
"""

from __future__ import annotations

from .loggers import LogEvent

# Summary key -> short console label, in the order they are printed.
_SUMMARY_LABELS: tuple[tuple[str, str], ...] = (
    ("Summary/success_rate", "success"),
    ("Summary/success_rate_stochastic", "success_stoch"),
    ("Summary/expert_success_rate", "success_expert"),
    ("Summary/return_per_step", "return/step"),
    ("Summary/best_return_per_step", "best_return/step"),
    ("Summary/feature_gap_norm", "feature_gap"),
    ("Summary/reward_loss", "reward_loss"),
    ("Summary/reward_weight_cosine", "reward_cos"),
    ("Summary/critic_loss", "critic_loss"),
    ("Summary/q_mean", "q_mean"),
    ("Summary/policy_std", "policy_std"),
    ("Summary/actor_kl", "actor_kl"),
    ("Summary/policy_updates", "policy_updates"),
    ("Summary/train_reward_per_step", "train_reward/step"),
    ("Summary/train_episode_success_rate", "train_success"),
    ("Summary/training_episodes", "train_episodes"),
    ("Summary/environment_transitions", "total_interactions"),
    ("Summary/bc_loss", "bc_loss"),
    ("Summary/bc_val_loss", "bc_val_loss"),
)


class ConsoleLogger:
    """Print one concise line per training event."""

    def start(self, num_iterations: int) -> None:
        print(
            f"[INFO] Starting learning for {int(num_iterations)} iterations.",
            flush=True,
        )

    def finish(self, exit_code: int = 0) -> None:
        del exit_code

    def log(self, event: LogEvent) -> None:
        if event.kind == "iteration_summary":
            self._print_summary("iter", event)
        elif event.kind == "validation/summary":
            self._print_summary("validation", event)
        # Per-feature detail stays out of the terminal; it is a dashboard view.

    @staticmethod
    def _print_summary(label: str, event: LogEvent) -> None:
        fields = [
            f"{name}={event.payload[key]:.4g}"
            for key, name in _SUMMARY_LABELS
            if key in event.payload
        ]
        if not fields:
            return
        transitions = event.payload.get("Summary/transitions")
        step = f"  {int(transitions):,} transitions" if transitions else ""
        print(
            f"[INFO] {label} {event.iteration + 1}{step}  " + "  ".join(fields),
            flush=True,
        )
