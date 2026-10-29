"""Publish checkpoint evaluations as directly comparable W&B streams.

Each learner gets identical metric keys and explicit environment-interaction
axes. A report overlays the streams. Publishing only reads saved evaluations;
it never launches, resets, or modifies a learner.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from config.cli import build_evaluation_dashboard_parser
from data.checkpoint_evaluations import TRAINING_INTERACTIONS, checkpoint_events

_COLORS = (
    "#2563eb",
    "#ea580c",
    "#16a34a",
    "#9333ea",
    "#db2777",
    "#0891b2",
    "#854d0e",
    "#475569",
    "#dc2626",
    "#65a30d",
    "#7c3aed",
    "#0d9488",
    "#c2410c",
)
_METRICS = (
    ("reward_per_step", "Task reward per environment step"),
    ("sustained_success_rate", "Sustained goal success (final 25 steps)"),
    ("lift_rate", "Lifted at least 10 cm"),
)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def _label(state: dict[str, Any], job: dict[str, Any]) -> str:
    if "label" in job:
        return job["label"]
    quantile = state["protocol"].get("control_quantiles", {}).get(job["arm"])
    name = (
        f"Successor PMD q={quantile:.2f}"
        if quantile is not None
        else job["algorithm"].upper()
    )
    return f"{name} · seed {job['seed']}"


def _publish_job(state, job, entry, directory, entity, project) -> bool:
    import wandb

    pending = [
        event
        for event in checkpoint_events(job)
        if event["id"] not in entry["published"]
    ]
    if not pending:
        return False
    with wandb.init(
        entity=entity,
        project=project,
        id=entry["run_id"],
        resume="allow",
        name=_label(state, job),
        group=state["experiment"] + "_evaluations",
        job_type="checkpoint-evaluation",
        tags=["evaluation", "sample-efficiency"],
        dir=str(directory),
        config={
            "source_experiment": state["experiment"],
            "source_learner": job["name"],
            "algorithm": job["algorithm"],
            "seed": job["seed"],
            "control_quantile": state["protocol"]
            .get("control_quantiles", {})
            .get(job["arm"]),
            "x_axis": "physical training environment interactions, summed across all lanes",
            "evaluation_protocol": state["protocol"]["evaluation"],
            **job.get("parameters", {}),
        },
        settings=wandb.Settings(console="off", disable_code=True, x_disable_stats=True),
    ) as run:
        run.define_metric("Samples/*", hidden=True)
        run.define_metric(
            "Evaluation/*", step_metric=TRAINING_INTERACTIONS, step_sync=False
        )
        for event in pending:
            run.log(event["payload"])
        run.summary["source_status"] = job["status"]
        entry["url"] = run.url
    # Finish flushes this writer before progress is recorded or it is reopened.
    entry["published"].extend(event["id"] for event in pending)
    print(f"Published {len(pending)} checkpoints: {_label(state, job)}", flush=True)
    return True


def build_report_blocks(state: dict[str, Any], manifest: dict[str, Any]) -> list:
    """Overlay all learners with explicit axes and identical y metric names."""
    import wandb_workspaces.reports.v2 as wr

    entries = manifest["runs"]
    runset = wr.Runset(
        name="Evaluation comparisons",
        filters=" or ".join(f'ID == "{entry["run_id"]}"' for entry in entries.values()),
        run_settings={
            entry["run_id"]: wr.RunSettings(color=_COLORS[index % len(_COLORS)])
            for index, entry in enumerate(entries.values())
        },
    )
    latest = [
        max(
            (
                result["training_environment_transitions"]
                for result in job["evaluations"]
                if result["variant"] == "lift_easy"
            ),
            default=0,
        )
        for job in state["jobs"]
    ]
    evaluated_limits = [count for count in latest if count > 0]
    common_limit = min(evaluated_limits) if len(evaluated_limits) >= 2 else None

    metrics = _METRICS
    if state.get("protocol", {}).get("include_failure_diagnostics"):
        metrics = (*_METRICS, ("early_termination_rate", "Early failure rate"))

    def grid(variant: str, *, upper: int | None = None, cost_axis: bool = False):
        panels = []
        for mode in ("deterministic", "stochastic"):
            for metric, title in metrics:
                index = len(panels)
                panels.append(
                    wr.LinePlot(
                        title=f"{title} — {mode}",
                        x=(
                            f"Samples/{variant}/total_environment_interactions"
                            if cost_axis
                            else TRAINING_INTERACTIONS
                        ),
                        y=[f"Evaluation/{variant}/{mode}/{metric}"],
                        title_x=(
                            "Environment interactions (training + evaluation)"
                            if cost_axis
                            else "Training environment interactions"
                        ),
                        title_y="Reward / step"
                        if metric == "reward_per_step"
                        else "Fraction of episodes",
                        range_x=(0, upper) if upper else (None, None),
                        range_y=(None, None) if metric == "reward_per_step" else (0, 1),
                        log_x=upper is None,
                        smoothing_type="none",
                        smoothing_factor=0,
                        aggregate=False,
                        max_runs_to_show=len(entries),
                        legend_position="south",
                        legend_template="${run:name}",
                        layout=wr.Layout(
                            x=(index % len(metrics)) * (24 // len(metrics)),
                            y=(index // len(metrics)) * 6,
                            w=24 // len(metrics),
                            h=6,
                        ),
                    )
                )
        return wr.PanelGrid(runsets=[runset], panels=panels)

    blocks = [
        wr.P(
            state.get("protocol", {}).get(
                "report_description",
                "Each chart overlays PPO and all three successor-PMD quantiles. X counts physical "
                "environment interactions across all vector lanes. Y is held-out task performance; "
                "entropy bonuses are excluded. The upper row uses the deterministic policy; the "
                "lower row samples actions. Each point uses 64 episodes per mode (32 for each "
                "evaluation seed). This comparison has one training seed. Lines are unsmoothed; "
                "there is no extrapolation beyond observed checkpoints.",
            )
        )
    ]
    if common_limit:
        blocks += [
            wr.H1(f"Same training budget: 0–{common_limit:,} interactions"),
            wr.P(
                "Linear x-axis, clipped to the latest evaluated training budget available "
                "for every evaluated learner. Queued trials have no curve yet. "
                "No score is invented before a learner's first checkpoint."
            ),
            grid("lift_easy", upper=common_limit),
        ]
    blocks += [
        wr.H1("Full training range — easy Lift"),
        wr.P(
            "Logarithmic x-axis keeps the low-sample PMD region visible alongside PPO's "
            "longer run. Labels are actual interaction counts, not checkpoint indices."
        ),
        grid("lift_easy"),
        wr.H1("Including evaluation interaction costs — easy Lift"),
        wr.P(
            "X adds cumulative external evaluation interactions, including transfer checks. "
            "Evaluation data is not used for training. Logarithmic x-axis."
        ),
        grid("lift_easy", cost_axis=True),
    ]
    if any(
        result["variant"] == "standard"
        for job in state["jobs"]
        for result in job["evaluations"]
    ):
        blocks += [
            wr.H1("Transfer to standard resets"),
            wr.P(
                "Same policies, original reset distribution. Scores are separate from easy Lift. "
                "Logarithmic training-interaction axis."
            ),
            grid("standard"),
        ]
    return blocks


def _publish_report(state: dict[str, Any], manifest: dict[str, Any]) -> None:
    import wandb_workspaces.reports.v2 as wr

    report = (
        wr.Report.from_url(manifest["report_url"])
        if "report_url" in manifest
        else wr.Report(
            entity=manifest["entity"],
            project=manifest["project"],
            title=state["protocol"].get(
                "report_title", "Lift: PPO vs successor PMD — sample efficiency"
            ),
            description=f"Checkpoint evaluations for {state['experiment']}",
            width="fluid",
        )
    )
    report.blocks = build_report_blocks(state, manifest)
    report.save()
    manifest["report_url"] = report.url
    # Keep the old diagnostic run useful as an entry point to the comparison.
    if state.get("wandb_url"):
        import wandb

        source_id = state["wandb_url"].rstrip("/").rsplit("/", 1)[-1]
        source = wandb.Api().run(
            f"{manifest['entity']}/{manifest['project']}/{source_id}"
        )
        link = f"Comparison report (all algorithms, environment-interaction axes): {report.url}"
        if link not in (source.notes or ""):
            source.notes = ((source.notes or "") + "\n\n" + link).strip()
            source.update()
        source.summary["comparison_report_url"] = report.url
        source.summary.update()
    print(f"Comparison report: {report.url}", flush=True)


def publish_evaluations(
    status_file: Path, *, entity: str, project: str, watch: bool, poll_seconds: float
) -> None:
    """Backfill and optionally follow an atomic comparison status.json file."""
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be finite and > 0.")
    directory = status_file.parent
    manifest_path = directory / "evaluation_dashboard.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"entity": entity, "project": project, "runs": {}}
    )
    if (manifest["entity"], manifest["project"]) != (entity, project):
        raise ValueError("Dashboard already exists in a different W&B project.")
    first_snapshot = True
    while True:
        state = json.loads(status_file.read_text())
        changed = False
        for job in state["jobs"]:
            entry = manifest["runs"].setdefault(
                job["name"],
                {
                    "run_id": "eval"
                    + hashlib.sha256(
                        f"{state['experiment']}/{job['name']}".encode()
                    ).hexdigest()[:12],
                    "published": [],
                },
            )
            changed |= _publish_job(state, job, entry, directory, entity, project)
            _write_manifest(manifest_path, manifest)
        if first_snapshot or changed or "report_url" not in manifest:
            _publish_report(state, manifest)
        first_snapshot = False
        manifest["source_status"] = state["status"]
        _write_manifest(manifest_path, manifest)
        if not watch or state["status"] not in {"starting", "running"}:
            break
        time.sleep(poll_seconds)


def main() -> None:
    from dotenv import load_dotenv

    args = build_evaluation_dashboard_parser().parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    publish_evaluations(
        Path(args.status_file).resolve(),
        entity=args.entity,
        project=args.project,
        watch=args.watch,
        poll_seconds=args.poll_seconds,
    )
