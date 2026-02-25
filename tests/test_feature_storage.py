import pytest


torch = pytest.importorskip("torch")

from utils.runtime_context import RuntimeContext
from storage.feature_storage import FeatureBufCfg, FeatureTrajectoryBuffer


def _make_buffer(num_envs: int, feature_dim: int, gamma: float) -> FeatureTrajectoryBuffer:
    cfg = FeatureBufCfg(min_ep_len=1, store_dtype=torch.float32)
    ctx = RuntimeContext(num_envs=num_envs, feature_dim=feature_dim, device="cpu")
    return FeatureTrajectoryBuffer(cfg=cfg, ctx=ctx, gamma=gamma)


def test_vectorized_buffer_collects_complete_episodes():
    buffer = _make_buffer(num_envs=2, feature_dim=2, gamma=0.99)

    for step in range(4):
        feats = torch.tensor(
            [[float(step), 0.0], [float(step), 1.0]],
            dtype=torch.float32,
        )
        dones = torch.tensor([step in (1, 3), step in (2, 3)], dtype=torch.bool)
        buffer.add_step(feats, dones)

    assert len(buffer) == 4
    assert buffer.steps_stored == 8

    feats, mask, lengths = buffer.sample_episodes(batch_size=3, device="cpu")
    assert feats.ndim == 3
    assert mask.ndim == 2
    assert lengths.ndim == 1
    assert torch.equal(mask.sum(dim=1), lengths)


def test_add_episode_and_discounted_feature_returns():
    gamma = 0.5
    buffer = _make_buffer(num_envs=1, feature_dim=2, gamma=gamma)

    episode_feats = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )
    buffer.add_episode(episode_feats)

    discounted_feats, lengths = buffer.sample_discounted_feature_returns(batch_size=1, device="cpu")
    expected = torch.tensor(
        [[1.0 + gamma * 3.0 + (gamma**2) * 5.0, 2.0 + gamma * 4.0 + (gamma**2) * 6.0]],
        dtype=torch.float32,
    )

    assert lengths.shape == (1,)
    assert lengths.item() == 3
    assert discounted_feats.shape == (1, 2)
    assert torch.allclose(discounted_feats, expected)
