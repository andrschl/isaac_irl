from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_collector_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "src" / "collectors" / "robomimic_data_collector.py"
    module_name = "robomimic_data_collector_test_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_collector_writes_tensor_batches_and_flushes_by_frequency(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_collector_module()

    collector = module.RobomimicDataCollector(
        task_name="Isaac-Unit-Test-v0",
        output_dir=str(tmp_path),
        filename="demos.hdf5",
        num_demos=4,
        flush_freq=2,
    )
    collector.reset()

    collector.add("obs", torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
    collector.add("actions", torch.tensor([[0.1], [0.2]], dtype=torch.float32))
    collector.add("rewards", torch.tensor([0.5, 0.6], dtype=torch.float32))

    collector.flush(torch.tensor([0]))
    collector.flush(torch.tensor([1]))
    collector.close()

    with h5py.File(tmp_path / "demos.hdf5", "r") as file_handle:
        assert "data" in file_handle
        assert sorted(file_handle["data"].keys()) == ["demo_0", "demo_1"]
        demo_0 = file_handle["data"]["demo_0"]
        assert int(demo_0.attrs["num_samples"]) == 1
        assert tuple(demo_0["obs"].shape) == (1, 2)
        assert tuple(demo_0["actions"].shape) == (1, 1)
        assert tuple(demo_0["rewards"].shape) == (1,)


def test_collector_writes_nested_mapping_payload_per_env(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_collector_module()

    collector = module.RobomimicDataCollector(
        task_name="Isaac-Unit-Test-v0",
        output_dir=str(tmp_path),
        filename="demos.hdf5",
        num_demos=2,
        flush_freq=1,
    )
    collector.reset()

    collector.add(
        "obs",
        {
            "policy": {
                "joint_pos": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                "joint_vel": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
            },
        },
    )
    collector.add(
        "features",
        {
            "reach": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "lift": torch.tensor([1.0, 2.0], dtype=torch.float32),
        },
    )
    collector.add("dones", torch.tensor([True, True], dtype=torch.bool))

    collector.flush([0, 1])
    collector.close()

    with h5py.File(tmp_path / "demos.hdf5", "r") as file_handle:
        demo_0 = file_handle["data"]["demo_0"]
        assert "obs" in demo_0
        assert "features" in demo_0
        assert "policy" in demo_0["obs"]
        assert "joint_pos" in demo_0["obs"]["policy"]
        assert tuple(demo_0["obs"]["policy"]["joint_pos"].shape) == (1, 2)
        assert tuple(demo_0["features"]["reach"].shape) == (1,)


def test_collector_stops_at_num_demos_limit(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_collector_module()

    collector = module.RobomimicDataCollector(
        task_name="Isaac-Unit-Test-v0",
        output_dir=str(tmp_path),
        filename="demos.hdf5",
        num_demos=1,
        flush_freq=1,
    )
    collector.reset()

    collector.add("obs", torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    collector.add("dones", torch.tensor([True, True], dtype=torch.bool))
    collector.flush([0, 1])
    collector.close()

    with h5py.File(tmp_path / "demos.hdf5", "r") as file_handle:
        assert sorted(file_handle["data"].keys()) == ["demo_0"]


def test_collector_raises_on_batch_mismatch(tmp_path: Path):
    module = _load_collector_module()

    collector = module.RobomimicDataCollector(
        task_name="Isaac-Unit-Test-v0",
        output_dir=str(tmp_path),
        filename="demos.hdf5",
        num_demos=2,
        flush_freq=1,
    )
    collector.reset()

    collector.add("obs", torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    with pytest.raises(ValueError, match="Batch mismatch"):
        collector.add("actions", torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32))


def test_collector_raises_on_nested_batch_mismatch_with_leaf_path(tmp_path: Path):
    module = _load_collector_module()

    collector = module.RobomimicDataCollector(
        task_name="Isaac-Unit-Test-v0",
        output_dir=str(tmp_path),
        filename="demos.hdf5",
        num_demos=2,
        flush_freq=1,
    )
    collector.reset()

    collector.add(
        "obs",
        {
            "policy": {
                "joint_pos": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            }
        },
    )
    with pytest.raises(ValueError, match="obs/policy/joint_vel"):
        collector.add(
            "obs",
            {
                "policy": {
                    "joint_vel": torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32),
                }
            },
        )


def test_flush_deduplicates_env_ids(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_collector_module()

    collector = module.RobomimicDataCollector(
        task_name="Isaac-Unit-Test-v0",
        output_dir=str(tmp_path),
        filename="demos.hdf5",
        num_demos=4,
        flush_freq=1,
    )
    collector.reset()

    collector.add("obs", torch.tensor([[1.0], [2.0]], dtype=torch.float32))
    collector.add("dones", torch.tensor([True, True], dtype=torch.bool))
    collector.flush(torch.tensor([0, 0, 1, 1]))
    collector.close()

    with h5py.File(tmp_path / "demos.hdf5", "r") as file_handle:
        assert sorted(file_handle["data"].keys()) == ["demo_0", "demo_1"]
