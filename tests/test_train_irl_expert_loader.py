from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _load_train_irl_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "irl" / "train_irl.py"
    module_name = "train_irl_module_expert_loader"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_load_feature_episodes_accepts_pt_payload(tmp_path: Path):
    module = _load_train_irl_module()
    payload_path = tmp_path / "expert.pt"

    torch.save(
        {
            "episodes": [
                torch.ones(3, 2, dtype=torch.float32),
                torch.zeros(5, 2, dtype=torch.float32),
            ]
        },
        payload_path,
    )

    episodes = module.load_feature_episodes(str(payload_path), expected_feature_dim=2)
    assert len(episodes) == 2
    assert episodes[0].shape == (3, 2)
    assert episodes[1].shape == (5, 2)


def test_load_feature_episodes_accepts_named_hdf5_payload(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_train_irl_module()
    payload_path = tmp_path / "expert.hdf5"

    with h5py.File(payload_path, "w") as file_handle:
        data_group = file_handle.create_group("data")

        demo_0 = data_group.create_group("demo_0")
        demo_0_features = demo_0.create_group("features")
        demo_0_features.create_dataset("reach", data=torch.ones(3).numpy())
        demo_0_features.create_dataset("lift", data=(2.0 * torch.ones(3)).numpy())

        demo_1 = data_group.create_group("demo_1")
        demo_1_features = demo_1.create_group("features")
        demo_1_features.create_dataset("reach", data=torch.zeros(4).numpy())
        demo_1_features.create_dataset("lift", data=torch.ones(4).numpy())

    episodes = module.load_feature_episodes(str(payload_path), expected_feature_dim=2)
    assert len(episodes) == 2
    assert episodes[0].shape == (3, 2)
    assert episodes[1].shape == (4, 2)
    assert torch.allclose(episodes[0][:, 0], torch.ones(3))
    assert torch.allclose(episodes[0][:, 1], 2.0 * torch.ones(3))


def test_load_feature_episodes_rejects_legacy_hdf5_matrix_payload(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_train_irl_module()
    payload_path = tmp_path / "legacy.hdf5"

    with h5py.File(payload_path, "w") as file_handle:
        data_group = file_handle.create_group("data")
        demo_0 = data_group.create_group("demo_0")
        demo_0.create_dataset("features", data=torch.ones(3, 2).numpy())

    with pytest.raises(ValueError, match="legacy feature format|Regenerate demos"):
        module.load_feature_episodes(str(payload_path), expected_feature_dim=2)


def test_load_feature_episodes_rejects_non_feature_torch_payload(tmp_path: Path):
    module = _load_train_irl_module()
    payload_path = tmp_path / "bad.pt"
    torch.save({"obs": torch.randn(3, 2)}, payload_path)

    with pytest.raises(ValueError, match="must contain one of keys"):
        module.load_feature_episodes(str(payload_path), expected_feature_dim=2)


def test_load_feature_episodes_rejects_feature_dim_mismatch(tmp_path: Path):
    module = _load_train_irl_module()
    payload_path = tmp_path / "bad_dim.pt"
    torch.save([torch.randn(3, 3)], payload_path)

    with pytest.raises(ValueError, match="feature dim mismatch"):
        module.load_feature_episodes(str(payload_path), expected_feature_dim=2)
