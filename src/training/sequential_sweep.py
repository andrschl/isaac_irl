"""One-GPU, strictly sequential Lift PMD screening and seed replication.

Training is uninterrupted. Only after its process exits are immutable
checkpoints evaluated. No training restart can silently discard replay.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

from config import load_experiment_cfg
from config.cli import build_sequential_sweep_parser

ROOT = Path(__file__).resolve().parents[2]
TRAINING_BUDGET = 800_000
BATCH_INTERACTIONS = 8_000
MILESTONES = (80_000, 200_000, 400_000, 640_000, TRAINING_BUDGET)
SCREEN_SETTINGS = ((0.75, 1), (0.5, 2), (0.25, 1), (0.75, 2), (0.5, 1), (0.25, 2))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def exclusive_gpu(lock_path: Path):
    """Reject overlapping sweep controllers; never stop another user's process."""
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"GPU sweep lock is already held: {lock_path}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "0",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def source_fingerprint(out: Path) -> dict[str, str]:
    paths = sorted((ROOT / "src").rglob("*.py")) + [
        ROOT / "scripts/train.py",
        ROOT / "scripts/tune_pmd.py",
        ROOT / "scripts/tune_pmd_parallel.py",
        ROOT / "scripts/publish_evaluations.py",
        out / "base.yaml",
        out / "evaluate.py",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def mode_rates(raw: dict, mode: str) -> dict:
    rows = [row for row in raw["rows"] if row["mode"] == mode]
    episodes = sum(row["episodes"] for row in rows)
    if episodes <= 0:
        raise ValueError(f"No completed {mode} evaluation episodes.")
    rates = {
        key + "_rate": sum(row[key] for row in rows) / episodes
        for key in (
            "held_goal_last25",
            "ever_goal_success",
            "ever_lift_10cm",
            "terminated_episodes",
        )
    }
    rates["return_mean"] = (
        sum(row["return_mean"] * row["episodes"] for row in rows) / episodes
    )
    rates["episodes"] = episodes
    return rates


def ranking_key(job: dict) -> tuple[float, ...]:
    """Rank the final two easy checkpoints, equally weighting both policy modes.

    Lexicographic: sustained success, reaching goal, fewer failures, episode
    return. Per-step reward cannot win by shortening episodes through failure.
    """
    results = sorted(
        (r for r in job["evaluations"] if r["variant"] == "lift_easy"),
        key=lambda r: r["training_environment_transitions"],
    )
    if (
        not results
        or not TRAINING_BUDGET - BATCH_INTERACTIONS
        <= results[-1]["training_environment_transitions"]
        <= TRAINING_BUDGET
    ):
        raise ValueError(
            "Selection requires a completed evaluation within one batch of the training budget."
        )
    rows = [
        result[mode]
        for result in results[-2:]
        for mode in ("deterministic", "stochastic")
    ]

    def mean(key: str) -> float:
        return sum(row[key] for row in rows) / len(rows)

    return (
        mean("held_goal_last25_rate"),
        mean("ever_goal_success_rate"),
        -mean("terminated_episodes_rate"),
        mean("return_mean"),
    )


def job_spec(
    quantile: float, utd: float, improvements: int, seed: int, phase: str
) -> dict:
    arm = f"q{round(100 * quantile):02d}_utd{utd:g}_p{improvements}"
    return {
        "name": f"{arm}_s{seed}",
        "arm": arm,
        "algorithm": "pmd",
        "seed": seed,
        "label": f"PMD q={quantile:.2f} | UTD={utd:g} | P={improvements} | seed {seed}",
        "phase": phase,
        "status": "queued",
        "evaluations": [],
        "evaluation_failures": [],
        "evaluation_transitions": 0,
        "evaluation_episodes": 0,
        "parameters": {
            "control_quantile": quantile,
            "utd": utd,
            "policy_improvements_per_batch": improvements,
        },
    }


class SequentialSweep:
    def __init__(self, out: Path, state: dict, hashes: dict):
        self.out, self.state, self.hashes = out, state, hashes
        self.active_process = None
        self.stop_requested = False
        self.started = 0.0
        self.deadline = 0.0

    def persist(self) -> None:
        self.state["updated_utc"] = now()
        self.state["elapsed_seconds"] = (
            time.monotonic() - self.started if self.started else 0
        )
        write_json(self.out / "status.json", self.state)

    def check_stop(self) -> None:
        if (
            self.stop_requested
            or (self.out / "STOP").exists()
            or time.monotonic() >= self.deadline
        ):
            raise InterruptedError("Sweep stopped by request or its wall-clock budget.")

    def execute(
        self, command: list[str], log: Path, *, isaac: bool, timeout: float
    ) -> None:
        if self.active_process is not None:
            raise RuntimeError("A child is still active; refusing concurrent launch.")
        self.check_stop()
        if source_fingerprint(self.out) != self.hashes:
            raise RuntimeError(
                "Source changed during the sweep; refusing mixed implementations."
            )
        if isaac:
            # A completed app can take a few seconds to release its CUDA context.
            wait_until = min(self.deadline, time.monotonic() + 120)
            while gpu_compute_pids():
                self.check_stop()
                if time.monotonic() >= wait_until:
                    raise RuntimeError(
                        "GPU 0 has another compute process; refusing overlapping Isaac."
                    )
                time.sleep(2)
        print(now(), "START", log.name, flush=True)
        with log.open("a") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.active_process = process
        self.state["active_process"] = {
            "pid": process.pid,
            "isaac": isaac,
            "command": command,
            "log": str(log),
            "started_utc": now(),
        }
        launched = time.monotonic()
        try:
            while process.poll() is None:
                self.persist()
                self.check_stop()
                if time.monotonic() - launched >= timeout:
                    raise TimeoutError(f"Child exceeded {timeout:.0f}s: {log}")
                time.sleep(5)
            if process.returncode != 0:
                raise RuntimeError(f"Child exited {process.returncode}; inspect {log}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=20)
            self.active_process = None
            self.state["active_process"] = None
            self.persist()
        print(now(), "END", log.name, flush=True)

    def config_for(self, job: dict) -> Path:
        config = yaml.safe_load((self.out / "base.yaml").read_text())
        config.update(
            experiment_name=self.out.name,
            run_name=job["name"],
            logger="wandb",
            wandb_project=self.state["project"],
            seed=job["seed"],
            max_iterations=10000,
        )
        config["env"].update(num_envs=32, reward_curriculum=False)
        config["reward"].update(source="environment", fit_diagnostic=False)
        config["runner"].update(
            validation_interval=0,
            validation_interval_transitions=None,
            transition_budget=None,
            total_transition_budget=self.state.get("protocol", {}).get(
                "training_budget", TRAINING_BUDGET
            ),
            save_interval=5,
        )
        p = job["parameters"]
        config["algo"]["type"] = "pmd"
        pmd = config["algo"]["pmd"]
        pmd["policy_improvements_per_batch"] = p["policy_improvements_per_batch"]
        pmd["train"]["critic_updates_per_transition"] = p["utd"]
        pmd["value"]["type"] = "successor"
        pmd["value"]["successor"]["control_quantile"] = p["control_quantile"]
        path = self.out / (job["name"] + ".yaml")
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        resolved, _ = load_experiment_cfg(
            args=argparse.Namespace(
                config=str(path), task="Isaac-Lift-Cube-Franka-v0", set=[]
            ),
            config_root=ROOT / "configs",
        )
        if (
            resolved.algorithm.value.control_quantile != p["control_quantile"]
            or resolved.algorithm.train.critic_updates_per_transition != p["utd"]
            or resolved.algorithm.policy_improvements_per_batch
            != p["policy_improvements_per_batch"]
            or resolved.algorithm.entropy_rate != 0.006
            or resolved.gamma != 0.98
        ):
            raise ValueError(
                "Resolved trial config does not match the declared experiment."
            )
        job["config"] = str(path)
        return path

    def estimated_trial_seconds(self, job: dict) -> float:
        completed = [
            j
            for j in self.state["jobs"]
            if j.get("phase") != "reference" and "training_seconds" in j
        ]
        # Conservative initial estimate, then use measured isolated throughput.
        per_utd = max(
            (j["training_seconds"] / j["parameters"]["utd"] for j in completed),
            default=4200,
        )
        return 1.2 * per_utd * job["parameters"]["utd"] + 600

    def checkpoints(self, job: dict) -> list[tuple[int, Path]]:
        import torch

        protocol = self.state.get("protocol", {})
        budget = protocol.get("training_budget", TRAINING_BUDGET)
        milestones = protocol.get("evaluation_milestones", MILESTONES)
        found = []
        for path in Path(job["run_dir"]).glob("model_*.pt"):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            count = int(saved["interaction_counts"]["environment_transitions"])
            found.append((count, path))
        selected = []
        for milestone in milestones:
            candidates = sorted(
                (count, path) for count, path in found if count >= milestone
            )
            if not candidates and milestone == budget:
                candidates = sorted(
                    (count, path)
                    for count, path in found
                    if budget - BATCH_INTERACTIONS <= count <= budget
                )
                candidates = candidates[-1:]
            if not candidates:
                raise RuntimeError(
                    f"Missing checkpoint at {milestone} physical interactions."
                )
            if candidates[0] not in selected:
                selected.append(candidates[0])
        return selected

    def evaluate(self, job: dict, checkpoint: Path, variant: str) -> None:
        stem = f"{job['name']}_{checkpoint.stem}_{variant}"
        descriptor = self.out / f"{stem}.input.json"
        output = self.out / f"{stem}.evaluation.json"
        write_json(
            descriptor,
            {
                "algorithm": "pmd",
                "config": job["config"],
                "checkpoint": str(checkpoint),
                "num_envs": 32,
                "seeds": [1001, 1002],
                "variant": variant,
            },
        )
        self.execute(
            [
                sys.executable,
                "-u",
                str(self.out / "evaluate.py"),
                str(descriptor),
                str(output),
            ],
            self.out / f"{stem}.evaluation.log",
            isaac=True,
            timeout=600,
        )
        raw = json.loads(output.read_text())
        job["evaluation_transitions"] += raw["evaluation_transitions"]
        job["evaluation_episodes"] += raw["evaluation_episodes"]
        result = {
            key: raw[key]
            for key in (
                "iteration",
                "training_transitions",
                "training_environment_transitions",
                "training_episodes",
                "checkpoint",
            )
        }
        result.update(
            variant=variant,
            evaluation_file=str(output),
            cumulative_evaluation_transitions=job["evaluation_transitions"],
            cumulative_evaluation_episodes=job["evaluation_episodes"],
            total_interactions=raw["training_environment_transitions"]
            + job["evaluation_transitions"],
            deterministic=mode_rates(raw, "deterministic"),
            stochastic=mode_rates(raw, "stochastic"),
        )
        job["evaluations"].append(result)
        self.persist()

    def publish(self) -> None:
        self.execute(
            [
                sys.executable,
                "-u",
                "scripts/publish_evaluations.py",
                str(self.out / "status.json"),
                "--entity",
                self.state["entity"],
                "--project",
                self.state["project"],
            ],
            self.out / "dashboard.log",
            isaac=False,
            timeout=300,
        )

    def trial(self, job: dict) -> bool:
        if self.deadline - time.monotonic() < self.estimated_trial_seconds(job):
            job["status"] = "skipped_time_budget"
            self.persist()
            return False
        config = self.config_for(job)
        job.update(status="training", started_utc=now())
        self.persist()
        started = time.monotonic()
        command = [
            sys.executable,
            "-u",
            "scripts/train.py",
            "--config",
            str(config),
            "--task",
            "Isaac-Lift-Cube-Franka-v0",
            "--headless",
            "--logger",
            "wandb",
            "--algorithm",
            "pmd",
            "--value_backend",
            "successor",
            "--reward_source",
            "environment",
        ]
        job["command"] = command
        self.execute(
            command, self.out / f"{job['name']}.train.log", isaac=True, timeout=5 * 3600
        )
        job["training_seconds"] = time.monotonic() - started
        runs = list((ROOT / "logs/train" / self.out.name).glob("*_" + job["name"]))
        if len(runs) != 1:
            raise RuntimeError(
                f"Expected one run directory for {job['name']}, found {len(runs)}."
            )
        job.update(run_dir=str(runs[0]), status="evaluating")
        checkpoints = self.checkpoints(job)
        for _, checkpoint in checkpoints:
            self.evaluate(job, checkpoint, "lift_easy")
        self.evaluate(job, checkpoints[-1][1], "standard")
        job.update(status="complete", ended_utc=now(), ranking=list(ranking_key(job)))
        self.persist()
        self.publish()
        return True

    def ranked(self) -> list[dict]:
        return sorted(
            (
                j
                for j in self.state["jobs"]
                if j.get("phase") in {"screen", "cadence"} and j["status"] == "complete"
            ),
            key=ranking_key,
            reverse=True,
        )

    def run(self) -> None:
        self.started = time.monotonic()
        self.deadline = self.started + self.state["max_hours"] * 3600
        self.state.update(
            status="running", started_utc=now(), controller_pid=os.getpid()
        )
        self.persist()
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_requested", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_requested", True))
        try:
            self.publish()
            for job in list(self.state["jobs"]):
                if job.get("phase") == "screen":
                    self.trial(job)
            ranked = self.ranked()
            if not ranked:
                raise RuntimeError("No completed screening trials to select from.")
            best = ranked[0]["parameters"]
            self.state["screen_winner"] = ranked[0]["name"]
            for improvements in (16, 8):
                job = job_spec(
                    best["control_quantile"], best["utd"], improvements, 42, "cadence"
                )
                self.state["jobs"].append(job)
                self.trial(job)
            finalists = self.ranked()[:2]
            self.state["replication_finalists"] = [j["name"] for j in finalists]
            for seed in (43, 44):
                for finalist in finalists:
                    p = finalist["parameters"]
                    job = job_spec(
                        p["control_quantile"],
                        p["utd"],
                        p["policy_improvements_per_batch"],
                        seed,
                        "replication",
                    )
                    self.state["jobs"].append(job)
                    self.trial(job)
            self.state["status"] = "complete"
            self.persist()
            self.publish()
        except InterruptedError as error:
            self.state.update(status="stopped_by_request_or_watchdog", error=str(error))
        except BaseException as error:
            self.state.update(status="failed", error=repr(error))
            raise
        finally:
            for job in self.state["jobs"]:
                if job["status"] in {"training", "evaluating"}:
                    job["status"] = "incomplete"
            self.persist()


def prepare(args) -> tuple[Path, dict, dict]:
    out = Path(args.output_dir).resolve()
    if not out.is_relative_to(ROOT / "logs/benchmarks"):
        raise ValueError("Place sweep artifacts under logs/benchmarks/.")
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 24:
        raise ValueError("max_hours must be in (0, 24].")
    if (out / "status.json").exists():
        state = json.loads((out / "status.json").read_text())
        hashes = json.loads((out / "source_hashes.json").read_text())
        if state["status"] != "prepared":
            raise ValueError(
                "Use a fresh experiment directory; live replay cannot be resumed by this controller."
            )
        if hashes != source_fingerprint(out):
            raise ValueError(
                "Source changed since preparation; use a fresh experiment directory."
            )
        return out, state, hashes
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.base_config, out / "base.yaml")
    shutil.copyfile(args.evaluator, out / "evaluate.py")
    previous = json.loads(Path(args.reference_status).read_text())
    references = [copy.deepcopy(j) for j in previous["jobs"] if j["algorithm"] == "ppo"]
    for job in references:
        job.update(phase="reference", label=f"PPO reference | seed {job['seed']}")
    state = {
        "experiment": out.name,
        "status": "prepared",
        "max_hours": args.max_hours,
        "entity": args.entity,
        "project": args.project,
        "jobs": references
        + [job_spec(q, utd, 32, 42, "screen") for q, utd in SCREEN_SETTINGS],
        "protocol": {
            "report_title": "Lift PMD tuning — sequential GPU, equal interaction budgets",
            "report_description": "Sequential successor-PMD tuning; PPO reference is reused from the previous comparison. "
            "Legends give quantile, UTD, improvements per batch (P), and training seed. Each chart overlays "
            "settings on physical environment interactions. Upper row: deterministic; lower row: stochastic. "
            "Failures include every attempted episode. Training is uninterrupted; evaluation runs only after "
            "its learner exits. Replication seeds are individual curves, not additional hyperparameters.",
            "include_failure_diagnostics": True,
            "evaluation": "32 first complete episodes x seeds 1001/1002 x deterministic/stochastic",
            "training_budget": TRAINING_BUDGET,
            "evaluation_milestones": list(MILESTONES),
            "schedule": "six quantile/UTD trials; P=16/8 on winner; top two on seeds 43/44 if budget permits",
            "selection": "mean of final two checkpoints and both modes: sustained success, ever goal, fewer early failures, episode return (lexicographic)",
            "objective": "fixed task reward, gamma=.98, entropy_rate=.006, dt=.02, separate entropy critic",
            "source_reference": str(Path(args.reference_status).resolve()),
        },
    }
    hashes = source_fingerprint(out)
    write_json(out / "source_hashes.json", hashes)
    sweep = SequentialSweep(out, state, hashes)
    for job in state["jobs"]:
        if job["phase"] != "reference":
            sweep.config_for(job)
    sweep.persist()
    return out, state, hashes


def main() -> None:
    from dotenv import load_dotenv

    args = build_sequential_sweep_parser().parse_args()
    load_dotenv(ROOT / ".env", override=False)
    out, state, hashes = prepare(args)
    if args.prepare_only:
        print(f"Prepared {out}; six screening configurations validated.", flush=True)
        return
    os.environ.update(
        WANDB_MODE="online",
        WANDB_ENTITY=state["entity"],
        WANDB_RUN_GROUP=out.name,
        WANDB_DISABLE_CODE="true",
        PYTHONUNBUFFERED="1",
    )
    with exclusive_gpu(ROOT / "logs/benchmarks/.isaac_gpu0.lock"):
        SequentialSweep(out, state, hashes).run()
