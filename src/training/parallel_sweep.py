"""Two-learner training, followed by exclusive checkpoint evaluation.

Each training process owns its simulator, replay, RNGs, logs, and W&B run.
Only orchestration is parallel; learning and episode semantics are unchanged.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from training.sequential_sweep import (
    ROOT,
    SequentialSweep,
    exclusive_gpu,
    gpu_compute_pids,
    job_spec,
    now,
    source_fingerprint,
    write_json,
)


class ParallelSweep:
    """Train one seed-matched pair at a time, then evaluate with an idle GPU."""

    def __init__(self, shared: SequentialSweep):
        self.shared = shared
        self.children: dict[str, tuple[subprocess.Popen, dict, float]] = {}
        self.last_gpu_sample = 0.0

    def persist(self) -> None:
        self.shared.state["training_processes"] = [
            {"name": name, "pid": process.pid, "log": job["console_log"]}
            for name, (process, job, _) in self.children.items()
        ]
        self.shared.persist()

    def check_launch(self) -> None:
        self.shared.check_stop()
        if len(self.children) >= 2:
            raise RuntimeError("Two learners are already running; refusing a third.")
        if self.shared.active_process is not None:
            raise RuntimeError(
                "Evaluation/publication is active; cannot start training."
            )
        if source_fingerprint(self.shared.out) != self.shared.hashes:
            raise RuntimeError("Source changed; refusing mixed implementations.")
        owned = {process.pid for process, _, _ in self.children.values()}
        unexpected = set(gpu_compute_pids()) - owned
        if unexpected:
            raise RuntimeError(
                f"Unrelated GPU compute processes are active: {unexpected}."
            )

    def launch(self, job: dict) -> None:
        self.check_launch()
        config = self.shared.config_for(job)
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
        log = self.shared.out / f"{job['name']}.train.log"
        with log.open("a") as stream:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        job.update(
            status="training",
            pid=process.pid,
            command=command,
            console_log=str(log),
            started_utc=now(),
        )
        self.children[job["name"]] = (process, job, time.monotonic())
        self.persist()
        print(now(), "TRAIN START", job["name"], process.pid, flush=True)
        # Serialize app initialization/cache warmup, not learning. Each process
        # must reach its training loop before a second simulator is constructed.
        ready_deadline = time.monotonic() + 180
        while "[INFO] Starting learning" not in log.read_text():
            self.shared.check_stop()
            if process.poll() is not None:
                raise RuntimeError(
                    f"Learner exited during initialization; inspect {log}."
                )
            if time.monotonic() >= ready_deadline:
                raise TimeoutError(
                    f"Learner initialization exceeded 180s; inspect {log}."
                )
            self.persist()
            time.sleep(2)

    def sample_gpu(self) -> None:
        if time.monotonic() - self.last_gpu_sample < 30:
            return
        self.last_gpu_sample = time.monotonic()
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        utilization, memory, power = [
            float(v.strip()) for v in result.stdout.strip().split(",")
        ]
        sample = {
            "time_utc": now(),
            "utilization_percent": utilization,
            "memory_mib": memory,
            "power_watts": power,
            "learners": len(self.children),
        }
        with (self.shared.out / "gpu_usage.jsonl").open("a") as stream:
            stream.write(json.dumps(sample) + "\n")
        self.shared.state["gpu"] = sample

    def wait_training(self) -> None:
        while self.children:
            self.shared.check_stop()
            for name, (process, job, started) in list(self.children.items()):
                code = process.poll()
                if code is None:
                    continue
                job.update(exit_code=code, training_seconds=time.monotonic() - started)
                del self.children[name]
                if code != 0:
                    job.update(status="failed", ended_utc=now())
                    raise RuntimeError(
                        f"{name} exited {code}; inspect {job['console_log']}."
                    )
                runs = list(
                    (ROOT / "logs/train" / self.shared.out.name).glob("*_" + name)
                )
                if len(runs) != 1:
                    raise RuntimeError(
                        f"Expected one run directory for {name}, got {len(runs)}."
                    )
                job.update(status="awaiting_evaluation", run_dir=str(runs[0]))
                print(now(), "TRAIN END", name, flush=True)
            self.sample_gpu()
            self.persist()
            if self.children:
                time.sleep(5)

    def evaluate(self, job: dict) -> None:
        if self.children:
            raise RuntimeError(
                "Training is still active; evaluation must be exclusive."
            )
        job["status"] = "evaluating"
        self.persist()
        checkpoints = self.shared.checkpoints(job)
        for _, checkpoint in checkpoints:
            self.shared.evaluate(job, checkpoint, "lift_easy")
        self.shared.evaluate(job, checkpoints[-1][1], "standard")
        job.update(status="complete", ended_utc=now())
        self.persist()
        self.shared.publish()

    def stop_owned(self) -> None:
        for process, _, _ in self.children.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, job, _ in self.children.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)
            job.update(status="incomplete", exit_code=process.returncode)
        self.children.clear()

    def run(self) -> None:
        shared, state = self.shared, self.shared.state
        shared.started = time.monotonic()
        shared.deadline = shared.started + state["max_hours"] * 3600
        state.update(status="running", started_utc=now(), controller_pid=os.getpid())
        signal.signal(
            signal.SIGTERM, lambda *_: setattr(shared, "stop_requested", True)
        )
        signal.signal(signal.SIGINT, lambda *_: setattr(shared, "stop_requested", True))
        self.persist()
        try:
            shared.publish()
            wave_durations = []
            for seed in state["protocol"]["training_seeds"]:
                wave = [
                    j
                    for j in state["jobs"]
                    if j.get("phase") == "long_run" and j["seed"] == seed
                ]
                # First pair is attempted under the hard watchdog. Later pairs
                # require enough time at the measured shared-GPU throughput.
                if wave_durations and shared.deadline - time.monotonic() < 1.1 * max(
                    wave_durations
                ):
                    for job in wave:
                        job["status"] = "skipped_time_budget"
                    self.persist()
                    continue
                started = time.monotonic()
                for job in wave:
                    self.launch(job)
                self.wait_training()
                for job in wave:
                    self.evaluate(job)
                wave_durations.append(time.monotonic() - started)
            state["status"] = "complete"
            self.persist()
            shared.publish()
        except InterruptedError as error:
            state.update(status="stopped_by_request_or_watchdog", error=str(error))
        except BaseException as error:
            state.update(status="failed", error=repr(error))
            raise
        finally:
            self.stop_owned()
            for job in state["jobs"]:
                if job["status"] in {"training", "evaluating", "awaiting_evaluation"}:
                    job["status"] = "incomplete"
            self.persist()


def prepare(args) -> SequentialSweep:
    out = Path(args.output_dir).resolve()
    if (
        not out.is_relative_to(ROOT / "logs/benchmarks")
        or out == ROOT / "logs/benchmarks"
    ):
        raise ValueError("Use a fresh experiment subdirectory of logs/benchmarks/.")
    if not math.isfinite(args.max_hours) or not 0 < args.max_hours <= 48:
        raise ValueError("max_hours must be in (0, 48].")
    if args.training_budget < 800_000 or args.training_budget % 8000:
        raise ValueError("training_budget must be >=800000 and a multiple of 8000.")
    if (out / "status.json").exists():
        state = json.loads((out / "status.json").read_text())
        hashes = json.loads((out / "source_hashes.json").read_text())
        if state["status"] != "prepared" or hashes != source_fingerprint(out):
            raise ValueError(
                "Use a fresh directory; never resume a learner without its replay."
            )
        return SequentialSweep(out, state, hashes)
    previous_path = Path(args.reference_status).resolve()
    previous = json.loads(previous_path.read_text())
    if previous["status"] != "complete":
        raise ValueError(
            "Let the previous sweep finish before launching parallel follow-ups."
        )
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(previous_path.parent / "base.yaml", out / "base.yaml")
    shutil.copyfile(previous_path.parent / "evaluate.py", out / "evaluate.py")
    references = [copy.deepcopy(j) for j in previous["jobs"] if j["algorithm"] == "ppo"]
    for job in references:
        job["phase"] = "reference"
    finalists = [
        j for j in previous["jobs"] if j["name"] in previous["replication_finalists"]
    ]
    if len(finalists) != 2 or len(args.seeds) != len(set(args.seeds)) or not args.seeds:
        raise ValueError(
            "Require exactly two finalists and distinct nonempty training seeds."
        )
    jobs = []
    for seed in args.seeds:
        for finalist in finalists:
            p = finalist["parameters"]
            jobs.append(
                job_spec(
                    p["control_quantile"],
                    p["utd"],
                    p["policy_improvements_per_batch"],
                    seed,
                    "long_run",
                )
            )
    milestones = sorted(
        {
            80_000,
            200_000,
            400_000,
            800_000,
            args.training_budget // 2,
            3 * args.training_budget // 4,
            args.training_budget,
        }
    )
    state = {
        "experiment": out.name,
        "status": "prepared",
        "max_hours": args.max_hours,
        "entity": args.entity,
        "project": args.project,
        "jobs": references + jobs,
        "protocol": {
            "report_title": "Lift PMD — longer runs, two concurrent learners",
            "report_description": "Two concurrent successor-PMD learners; serialized app initialization and exclusive evaluation after both learners finish. "
            "Every run starts from scratch with uninterrupted replay. PPO is the prior same-objective reference. "
            "Legends specify quantile, UTD, improvements per batch (P), and seed; x counts physical interactions. "
            "Shared-GPU wall times are not isolated algorithm speed comparisons.",
            "include_failure_diagnostics": True,
            "evaluation": previous["protocol"]["evaluation"],
            "objective": previous["protocol"]["objective"],
            "training_budget": args.training_budget,
            "evaluation_milestones": milestones,
            "training_seeds": args.seeds,
            "max_concurrent_learners": 2,
            "source_reference": str(previous_path),
            "finalists": previous["replication_finalists"],
        },
    }
    hashes = source_fingerprint(out)
    write_json(out / "source_hashes.json", hashes)
    shared = SequentialSweep(out, state, hashes)
    for job in jobs:
        shared.config_for(job)
    shared.persist()
    return shared


def main() -> None:
    from dotenv import load_dotenv

    from config.cli import build_parallel_sweep_parser

    args = build_parallel_sweep_parser().parse_args()
    load_dotenv(ROOT / ".env", override=False)
    shared = prepare(args)
    if args.prepare_only:
        print(
            f"Prepared {shared.out}: two learners, {args.training_budget:,} interactions each."
        )
        return
    os.environ.update(
        WANDB_MODE="online",
        WANDB_ENTITY=args.entity,
        WANDB_RUN_GROUP=shared.out.name,
        WANDB_DISABLE_CODE="true",
        PYTHONUNBUFFERED="1",
    )
    with exclusive_gpu(ROOT / "logs/benchmarks/.isaac_gpu0.lock"):
        ParallelSweep(shared).run()
