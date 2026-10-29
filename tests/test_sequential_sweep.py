"""Sequential launch and selection contracts, without Isaac or W&B."""

import copy
import time

import pytest
import yaml

from training.sequential_sweep import (
    ROOT,
    SCREEN_SETTINGS,
    SequentialSweep,
    exclusive_gpu,
    job_spec,
    ranking_key,
)


def test_exclusive_gpu_refuses_overlap_and_releases(tmp_path):
    path = tmp_path / "gpu.lock"
    with exclusive_gpu(path):
        with pytest.raises(RuntimeError, match="already held"):
            with exclusive_gpu(path):
                pass
    with exclusive_gpu(path):
        pass


def test_child_must_exit_before_next_launch(tmp_path, monkeypatch):
    import training.sequential_sweep as module

    events = []

    class Child:
        pid = 123
        returncode = 0

        def poll(self):
            events.append("exit")
            return 0

    def launch(*args, **kwargs):
        if events:
            assert events[-1] == "exit"
        events.append("launch")
        return Child()

    sweep = SequentialSweep(tmp_path, {"active_process": None}, {})
    sweep.deadline = time.monotonic() + 60
    monkeypatch.setattr(module, "source_fingerprint", lambda _: {})
    monkeypatch.setattr(module, "gpu_compute_pids", lambda: [])
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    sweep.execute(["train"], tmp_path / "train.log", isaac=True, timeout=10)
    sweep.execute(["evaluate"], tmp_path / "eval.log", isaac=True, timeout=10)
    assert events.count("launch") == 2
    assert sweep.active_process is None
    sweep.active_process = Child()
    with pytest.raises(RuntimeError, match="still active"):
        sweep.execute(["train"], tmp_path / "other.log", isaac=True, timeout=10)


def test_selection_prefers_success_then_fewer_failures_over_reward():
    def job(success, failure, reward):
        rates = {
            "held_goal_last25_rate": success,
            "ever_goal_success_rate": success,
            "terminated_episodes_rate": failure,
            "return_mean": reward,
        }
        return {
            "evaluations": [
                {
                    "variant": "lift_easy",
                    "training_environment_transitions": 800000,
                    "deterministic": rates,
                    "stochastic": rates,
                }
            ]
        }

    assert ranking_key(job(0.5, 0.1, 1)) > ranking_key(job(0, 0, 100))
    assert ranking_key(job(0, 0, 1)) > ranking_key(job(0, 0.5, 100))
    partial = job(1, 0, 100)
    partial["evaluations"][0]["training_environment_transitions"] = 400000
    with pytest.raises(ValueError, match="training budget"):
        ranking_key(partial)


def test_watchdog_terminates_only_owned_process_and_waits(tmp_path, monkeypatch):
    import training.sequential_sweep as module

    events = []

    class Child:
        pid = 123
        returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            events.append("waited")
            self.returncode = -15

    checks = 0

    def check_stop():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise InterruptedError("watchdog")

    sweep = SequentialSweep(tmp_path, {}, {})
    sweep.deadline = time.monotonic() + 60
    monkeypatch.setattr(sweep, "check_stop", check_stop)
    monkeypatch.setattr(module, "source_fingerprint", lambda _: {})
    monkeypatch.setattr(module, "gpu_compute_pids", lambda: [])
    monkeypatch.setattr(module.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: events.append((pid, sig)))
    with pytest.raises(InterruptedError, match="watchdog"):
        sweep.execute(["train"], tmp_path / "train.log", isaac=True, timeout=60)
    assert events == [(123, module.signal.SIGTERM), "waited"]
    assert sweep.active_process is None
    assert sweep.state["active_process"] is None


@pytest.mark.parametrize("quantile,utd", SCREEN_SETTINGS)
def test_trial_config_preserves_objective_and_changes_only_declared_parameters(
    tmp_path, quantile, utd
):
    base = yaml.safe_load((ROOT / "configs/lift_cube_franka/pmd_easy.yaml").read_text())
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base))
    sweep = SequentialSweep(tmp_path, {"project": "isaac_irl"}, {})
    job = job_spec(quantile, utd, 32, 42, "screen")
    path = sweep.config_for(job)  # also strict-loads the actual launcher config
    actual = yaml.safe_load(path.read_text())
    expected_algorithm = copy.deepcopy(base["algo"])
    expected_algorithm["pmd"]["train"]["critic_updates_per_transition"] = utd
    expected_algorithm["pmd"]["value"]["successor"]["control_quantile"] = quantile
    assert actual["algo"] == expected_algorithm
    assert actual["env"] == base["env"]
    assert actual["reward"] == base["reward"]
    assert actual["runner"]["total_transition_budget"] == 800000
    assert actual["runner"]["validation_interval"] == 0
