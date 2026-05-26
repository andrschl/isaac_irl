from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Add project root so storage, utils are importable
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data import (
    load_expert_success_rate,
    load_feature_episodes,
    subset_episodes,
)


def _load_train_irl_module():
    """Compatibility shim — only used for tests that touch script-side helpers
    (e.g. ``_install_expert_episodes``) that still live in
    ``scripts/irl/train_irl.py``."""
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

    episodes = load_feature_episodes(str(payload_path), expected_feature_dim=2)
    assert len(episodes) == 2
    assert episodes[0].shape == (3, 2)
    assert episodes[1].shape == (5, 2)


def test_load_feature_episodes_accepts_named_hdf5_payload(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_train_irl_module()
    payload_path = tmp_path / "expert.hdf5"

    with h5py.File(payload_path, "w", track_order=True) as file_handle:
        data_group = file_handle.create_group("data", track_order=True)

        demo_0 = data_group.create_group("demo_0", track_order=True)
        demo_0_features = demo_0.create_group("features", track_order=True)
        demo_0_features.create_dataset("reach", data=torch.ones(3).numpy())
        demo_0_features.create_dataset("lift", data=(2.0 * torch.ones(3)).numpy())

        demo_1 = data_group.create_group("demo_1", track_order=True)
        demo_1_features = demo_1.create_group("features", track_order=True)
        demo_1_features.create_dataset("reach", data=torch.zeros(4).numpy())
        demo_1_features.create_dataset("lift", data=torch.ones(4).numpy())

    episodes = load_feature_episodes(
        str(payload_path),
        expected_feature_dim=2,
        expected_feature_names=["reach", "lift"],
    )
    assert len(episodes) == 2
    assert episodes[0].shape == (3, 2)
    assert episodes[1].shape == (4, 2)
    assert torch.allclose(episodes[0][:, 0], torch.ones(3))  # reach
    assert torch.allclose(episodes[0][:, 1], 2.0 * torch.ones(3))  # lift
    assert torch.allclose(episodes[1][:, 0], torch.zeros(4))  # reach
    assert torch.allclose(episodes[1][:, 1], torch.ones(4))  # lift


def test_load_feature_episodes_rejects_hdf5_feature_name_mismatch(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_train_irl_module()
    payload_path = tmp_path / "expert_bad_names.hdf5"

    with h5py.File(payload_path, "w", track_order=True) as file_handle:
        data_group = file_handle.create_group("data", track_order=True)
        demo_0 = data_group.create_group("demo_0", track_order=True)
        demo_0_features = demo_0.create_group("features", track_order=True)
        demo_0_features.create_dataset("reach", data=torch.ones(3).numpy())
        demo_0_features.create_dataset("lift", data=torch.ones(3).numpy())

    with pytest.raises(ValueError, match="feature-name mismatch"):
        load_feature_episodes(
            str(payload_path),
            expected_feature_dim=2,
            expected_feature_names=["reach", "success_bonus"],
        )


def test_load_expert_success_rate_any_time(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_train_irl_module()
    payload_path = tmp_path / "expert.hdf5"

    with h5py.File(payload_path, "w") as file_handle:
        data_group = file_handle.create_group("data")
        # demo_0 reaches goal at some step -> any-time success.
        d0 = data_group.create_group("demo_0")
        d0.create_dataset("success", data=torch.tensor([0.0, 1.0, 0.0]).numpy())
        # demo_1 never reaches goal.
        d1 = data_group.create_group("demo_1")
        d1.create_dataset("success", data=torch.tensor([0.0, 0.0]).numpy())

    rate = load_expert_success_rate(str(payload_path))
    assert rate == pytest.approx(0.5)


def test_load_expert_success_rate_none_when_absent(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_train_irl_module()
    payload_path = tmp_path / "expert_nosuccess.hdf5"

    with h5py.File(payload_path, "w") as file_handle:
        data_group = file_handle.create_group("data")
        demo_0 = data_group.create_group("demo_0")
        demo_0_features = demo_0.create_group("features")
        demo_0_features.create_dataset("reach", data=torch.ones(3).numpy())

    # No 'success' dataset on any demo -> None (back-compat with old demos).
    assert load_expert_success_rate(str(payload_path)) is None
    # Non-HDF5 payloads also return None.
    assert load_expert_success_rate(str(tmp_path / "x.pt")) is None


def test_load_feature_episodes_rejects_legacy_hdf5_matrix_payload(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_train_irl_module()
    payload_path = tmp_path / "legacy.hdf5"

    with h5py.File(payload_path, "w") as file_handle:
        data_group = file_handle.create_group("data")
        demo_0 = data_group.create_group("demo_0")
        demo_0.create_dataset("features", data=torch.ones(3, 2).numpy())

    with pytest.raises(ValueError, match="legacy feature format|Regenerate demos"):
        load_feature_episodes(str(payload_path), expected_feature_dim=2)


def test_load_feature_episodes_rejects_non_feature_torch_payload(tmp_path: Path):
    module = _load_train_irl_module()
    payload_path = tmp_path / "bad.pt"
    torch.save({"obs": torch.randn(3, 2)}, payload_path)

    with pytest.raises(ValueError, match="must contain one of keys"):
        load_feature_episodes(str(payload_path), expected_feature_dim=2)


def test_load_feature_episodes_rejects_feature_dim_mismatch(tmp_path: Path):
    module = _load_train_irl_module()
    payload_path = tmp_path / "bad_dim.pt"
    torch.save([torch.randn(3, 3)], payload_path)

    with pytest.raises(ValueError, match="feature dim mismatch"):
        load_feature_episodes(str(payload_path), expected_feature_dim=2)


def test_subset_episodes_first_returns_first_n():
    module = _load_train_irl_module()
    episodes = [
        torch.ones(2, 2),
        torch.ones(3, 2) * 2,
        torch.ones(4, 2) * 3,
    ]
    out = subset_episodes(episodes, max_num=2, strategy="first", seed=42)
    assert len(out) == 2
    assert torch.allclose(out[0], episodes[0])
    assert torch.allclose(out[1], episodes[1])


def test_subset_episodes_random_same_seed_returns_same_subset():
    module = _load_train_irl_module()
    episodes = [torch.ones(i + 1, 2) * i for i in range(10)]
    out_a = subset_episodes(episodes, max_num=3, strategy="random", seed=123)
    out_b = subset_episodes(episodes, max_num=3, strategy="random", seed=123)
    assert len(out_a) == 3
    assert len(out_b) == 3
    for a, b in zip(out_a, out_b):
        assert torch.allclose(a, b)


def test_install_expert_episodes_with_max_num_trajectories_loads_subset(tmp_path: Path):
    module = _load_train_irl_module()
    payload_path = tmp_path / "expert.pt"
    torch.save(
        {"episodes": [torch.ones(2, 2), torch.ones(3, 2), torch.ones(4, 2)]},
        payload_path,
    )

    from algorithms.irl import FeatureRewardLearner, IRLCfg
    from reward_model import LinearFeatureRewardModel, RewardModelCfg
    from storage.feature_storage import FeatureBufCfg
    from utils.runtime_context import RuntimeContext

    reward_model = LinearFeatureRewardModel(RewardModelCfg(num_features=2, is_linear=True))
    irl_alg = FeatureRewardLearner(reward=reward_model, gamma=0.99, cfg=IRLCfg(batch_size=2))
    irl_alg.init_expert_storage(
        runtime_ctx=RuntimeContext(num_envs=1, feature_dim=2),
        cfg=FeatureBufCfg(min_ep_len=1),
        num_envs=1,
    )

    module._install_expert_episodes(
        irl_alg,
        str(payload_path),
        expected_feature_dim=2,
        max_num_trajectories=1,
        subset_strategy="first",
        seed=42,
    )
    assert len(irl_alg.expert_storage) == 1
