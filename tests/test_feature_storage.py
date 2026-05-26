import pytest


torch = pytest.importorskip("torch")

from utils.runtime_context import RuntimeContext
from storage.feature_storage import FeatureBufCfg, FeatureTrajectoryBuffer


def _make_buffer(
    num_envs: int,
    feature_dim: int,
    gamma: float,
    *,
    min_ep_len: int = 1,
    capacity_steps: int = 1_000_000,
) -> FeatureTrajectoryBuffer:
    cfg = FeatureBufCfg(
        min_ep_len=min_ep_len,
        store_dtype=torch.float32,
        capacity_steps=capacity_steps,
    )
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

    discounted_feats, lengths = buffer.sample_discounted_feature_returns(
        batch_size=1, device="cpu"
    )
    expected = torch.tensor(
        [[1.0 + gamma * 3.0 + (gamma**2) * 5.0, 2.0 + gamma * 4.0 + (gamma**2) * 6.0]],
        dtype=torch.float32,
    )

    assert lengths.shape == (1,)
    assert lengths.item() == 3
    assert discounted_feats.shape == (1, 2)
    assert torch.allclose(discounted_feats, expected)


def test_finalize_in_progress_episodes_flushes_unfinished_streams():
    """Validation rolls out for a fixed horizon and may not see natural dones;
    finalize_in_progress_episodes must promote each env's pending stream into
    the completed-episodes deque."""
    buffer = _make_buffer(num_envs=2, feature_dim=1, gamma=1.0)
    for step in range(3):
        feats = torch.tensor([[float(step)], [float(step) + 10]], dtype=torch.float32)
        dones = torch.tensor([False, False])
        buffer.add_step(feats, dones)

    assert len(buffer) == 0  # nothing committed yet
    buffer.finalize_in_progress_episodes()
    assert len(buffer) == 2
    assert buffer.steps_stored == 6


def test_min_ep_len_drops_short_episodes():
    """Episodes shorter than ``min_ep_len`` must be discarded on commit."""
    buffer = _make_buffer(num_envs=1, feature_dim=1, gamma=1.0, min_ep_len=3)

    # 2-step episode (done at step 1) — should be dropped (length 2 < min_ep_len 3).
    buffer.add_step(torch.tensor([[1.0]]), torch.tensor([False]))
    buffer.add_step(torch.tensor([[2.0]]), torch.tensor([True]))
    assert len(buffer) == 0
    assert buffer.steps_stored == 0

    # 3-step episode (done at step 2) — should be kept.
    for step in range(3):
        done = torch.tensor([step == 2])
        buffer.add_step(torch.tensor([[float(step + 10)]]), done)
    assert len(buffer) == 1
    assert buffer.steps_stored == 3


def test_capacity_evicts_oldest_completed_episodes():
    """``capacity_steps`` caps total stored steps by FIFO-evicting whole episodes."""
    buffer = _make_buffer(num_envs=1, feature_dim=1, gamma=1.0, capacity_steps=4)

    def _commit_episode(value: float, length: int) -> None:
        for step in range(length):
            done = torch.tensor([step == length - 1])
            buffer.add_step(torch.tensor([[value]]), done)

    _commit_episode(1.0, length=3)
    _commit_episode(2.0, length=2)
    # 5 steps total > capacity 4 → oldest (length-3) episode evicted.
    assert len(buffer) == 1
    assert buffer.steps_stored == 2

    _commit_episode(3.0, length=2)
    # 4 steps total = capacity. Two length-2 episodes remain (values 2, 3).
    assert len(buffer) == 2
    assert buffer.steps_stored == 4
