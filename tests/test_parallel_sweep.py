"""Parallel launch bounds, ownership, and evaluation isolation without Isaac."""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from training import parallel_sweep as module
from training.sequential_sweep import SequentialSweep


def controller(tmp_path, monkeypatch):
    shared = SequentialSweep(tmp_path, {}, {})
    shared.deadline = time.monotonic() + 60
    monkeypatch.setattr(module, "source_fingerprint", lambda _: {})
    monkeypatch.setattr(module, "gpu_compute_pids", lambda: [])
    return module.ParallelSweep(shared)


def test_owned_gpu_peer_is_allowed_but_third_learner_is_rejected(tmp_path, monkeypatch):
    sweep = controller(tmp_path, monkeypatch)
    sweep.children["first"] = (SimpleNamespace(pid=123), {}, 0)
    monkeypatch.setattr(module, "gpu_compute_pids", lambda: [123])
    sweep.check_launch()
    sweep.children["second"] = (SimpleNamespace(pid=456), {}, 0)
    with pytest.raises(RuntimeError, match="third"):
        sweep.check_launch()


def test_foreign_gpu_process_is_not_allowed_or_killed(tmp_path, monkeypatch):
    sweep = controller(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "gpu_compute_pids", lambda: [999])
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda *args: pytest.fail("Must not kill foreign processes"),
    )
    with pytest.raises(RuntimeError, match="Unrelated"):
        sweep.check_launch()


def test_evaluation_cannot_overlap_training(tmp_path, monkeypatch):
    sweep = controller(tmp_path, monkeypatch)
    sweep.children["first"] = (SimpleNamespace(pid=123), {}, 0)
    with pytest.raises(RuntimeError, match="exclusive"):
        sweep.evaluate({})
    sweep.children.clear()
    sweep.shared.active_process = object()
    with pytest.raises(RuntimeError, match="Evaluation/publication"):
        sweep.check_launch()


def test_changed_source_and_expired_budget_block_launch(tmp_path, monkeypatch):
    sweep = controller(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "source_fingerprint", lambda _: {"file": "changed"})
    with pytest.raises(RuntimeError, match="Source changed"):
        sweep.check_launch()
    sweep.shared.deadline = 0
    with pytest.raises(InterruptedError):
        sweep.check_launch()


def test_shutdown_stops_both_owned_children_and_clears_state(tmp_path, monkeypatch):
    sweep = controller(tmp_path, monkeypatch)
    killed = []

    class Child:
        returncode = None

        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = -15

    jobs = [{"name": "first"}, {"name": "second"}]
    for pid, job in enumerate(jobs, 123):
        sweep.children[job["name"]] = (Child(pid), job, 0)
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: killed.append(pid))
    sweep.stop_owned()
    assert killed == [123, 124]
    assert not sweep.children
    assert all(job["status"] == "incomplete" for job in jobs)


def test_failure_stops_pair_without_evaluating_partial_training(tmp_path, monkeypatch):
    sweep = controller(tmp_path, monkeypatch)
    job = {"name": "first", "console_log": "first.log"}
    sweep.children["first"] = (SimpleNamespace(poll=lambda: 1), job, time.monotonic())
    with pytest.raises(RuntimeError, match="exited 1"):
        sweep.wait_training()
    assert job["status"] == "failed"
    assert not sweep.children


def test_two_launches_have_separate_logs_and_processes(tmp_path, monkeypatch):
    sweep = controller(tmp_path, monkeypatch)
    starts = []
    monkeypatch.setattr(
        sweep.shared, "config_for", lambda job: tmp_path / (job["name"] + ".yaml")
    )

    def launch(command, **kwargs):
        assert kwargs["start_new_session"]
        kwargs["stdout"].write("[INFO] Starting learning for 10000 iterations.\n")
        starts.append(command)
        return SimpleNamespace(pid=123 + len(starts), poll=lambda: None)

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    sweep.launch({"name": "p16"})
    sweep.launch({"name": "p32"})
    assert len(starts) == len(sweep.children) == 2
    processes = sweep.shared.state["training_processes"]
    assert len({p["pid"] for p in processes}) == 2
    assert len({p["log"] for p in processes}) == 2
    assert all(Path(p["log"]).exists() for p in processes)


def test_long_run_budget_preserves_learning_configuration(tmp_path):
    base = yaml.safe_load(
        (module.ROOT / "configs/lift_cube_franka/pmd_easy.yaml").read_text()
    )
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    shared = SequentialSweep(
        tmp_path, {"project": "isaac_irl", "protocol": {"training_budget": 3200000}}, {}
    )
    job = module.job_spec(0.5, 2, 16, 42, "long_run")
    actual = yaml.safe_load(shared.config_for(job).read_text())
    assert actual["runner"]["total_transition_budget"] == 3200000
    assert actual["runner"]["validation_interval"] == 0
    assert actual["env"] == base["env"]
    assert actual["reward"] == base["reward"]
    pmd = actual["algo"]["pmd"]
    assert pmd["entropy_rate"] == 0.006
    assert pmd["policy_improvements_per_batch"] == 16
    assert pmd["train"] == base["algo"]["pmd"]["train"]
    assert pmd["value"]["successor"]["control_quantile"] == 0.5


@pytest.mark.parametrize("final_count", [3200000, 3196000])
def test_long_checkpoint_selection_uses_declared_budget(tmp_path, final_count):
    import torch

    milestones = [80000, 200000, 400000, 800000, 1600000, 2400000, 3200000]
    counts = milestones[:-1] + [final_count]
    for count in counts:
        torch.save(
            {"interaction_counts": {"environment_transitions": count}},
            tmp_path / f"model_{count}.pt",
        )
    shared = SequentialSweep(
        tmp_path,
        {"protocol": {"training_budget": 3200000, "evaluation_milestones": milestones}},
        {},
    )
    assert [
        count for count, _ in shared.checkpoints({"run_dir": str(tmp_path)})
    ] == counts
