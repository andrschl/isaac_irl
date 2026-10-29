# ruff: noqa: E402
"""Storing and sampling the feature episodes a reward is learned from."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from config.schema import FeatureBufCfg
from storage.feature_storage import FeatureTrajectoryBuffer


def _buffer(feature_dim: int = 2, gamma: float = 0.5, **cfg) -> FeatureTrajectoryBuffer:
    settings = {"min_ep_len": 1, "store_dtype": torch.float32}
    settings.update(cfg)
    return FeatureTrajectoryBuffer(
        FeatureBufCfg(**settings), feature_dim=feature_dim, gamma=gamma, device="cpu"
    )


def test_a_generation_of_episodes_is_stored_in_one_call() -> None:
    buffer = _buffer()
    feats = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[True, True, True], [True, True, False]])

    buffer.replace_episode_batch(feats, mask)

    assert len(buffer) == 2
    assert buffer.steps_stored == 5


def test_replacing_drops_what_the_previous_generation_left() -> None:
    """Imitator features describe the current policy, never an older one."""
    buffer = _buffer()
    buffer.replace_episode_batch(
        torch.zeros(4, 3, 2), torch.ones(4, 3, dtype=torch.bool)
    )

    buffer.replace_episode_batch(
        torch.ones(1, 2, 2), torch.ones(1, 2, dtype=torch.bool)
    )

    assert len(buffer) == 1
    assert buffer.steps_stored == 2


def test_episodes_shorter_than_the_minimum_are_skipped() -> None:
    buffer = _buffer(min_ep_len=3)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    buffer.replace_episode_batch(torch.zeros(2, 3, 2), mask)

    assert len(buffer) == 1


def test_a_ragged_mask_is_refused() -> None:
    buffer = _buffer()
    gap = torch.tensor([[True, False, True]])

    with pytest.raises(ValueError, match="contiguous valid prefix"):
        buffer.replace_episode_batch(torch.zeros(1, 3, 2), gap)


def test_sampling_pads_to_the_longest_episode_and_says_which_steps_are_real() -> None:
    buffer = _buffer()
    buffer.append_episode(torch.ones(3, 2))
    buffer.append_episode(torch.ones(1, 2))

    feats, mask, lengths = buffer.sample_episodes(batch_size=8, device="cpu")

    assert feats.shape[0] == mask.shape[0] == lengths.shape[0] == 8
    assert feats.shape[1] == mask.shape[1]
    assert torch.equal(mask.sum(dim=1), lengths)
    assert bool(((lengths == 1) | (lengths == 3)).all())


def test_discounted_returns_stop_where_the_episode_did() -> None:
    gamma = 0.5
    buffer = _buffer(gamma=gamma)
    buffer.append_episode(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    )

    feats, mask, _ = buffer.sample_episodes(batch_size=1, device="cpu")
    returns = buffer.discounted_feature_returns(feats, mask, gamma)

    expected = torch.tensor(
        [[1.0 + gamma * 3.0 + gamma**2 * 5.0, 2.0 + gamma * 4.0 + gamma**2 * 6.0]]
    )
    torch.testing.assert_close(returns, expected)


def test_capacity_evicts_the_oldest_episodes_first() -> None:
    buffer = _buffer(capacity_steps=4)

    for _ in range(4):
        buffer.append_episode(torch.ones(2, 2))

    assert buffer.steps_stored == 4
    assert len(buffer) == 2


def test_sampling_an_empty_buffer_says_so() -> None:
    with pytest.raises(RuntimeError, match="no completed episodes"):
        _buffer().sample_episodes(batch_size=1, device="cpu")


def test_a_rollout_is_split_into_the_episodes_it_contains() -> None:
    """Discounting across a reset mixes the end of one episode with the next."""
    buffer = _buffer()
    features = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    dones = torch.tensor([[False, True, False, False, True, False]])

    buffer.add_rollout(features, dones)

    # Episodes of length 2 and 3, plus the trailing segment the horizon cut.
    assert len(buffer) == 3
    assert buffer.steps_stored == 6


def test_a_rollout_without_any_reset_is_one_episode() -> None:
    buffer = _buffer()

    buffer.add_rollout(torch.zeros(2, 4, 2), torch.zeros(2, 4, dtype=torch.bool))

    assert len(buffer) == 2, "one trailing segment per environment"
    assert buffer.steps_stored == 8


def test_returns_stop_at_a_reset_rather_than_discounting_through_it() -> None:
    gamma = 0.5
    buffer = _buffer(gamma=gamma)
    # Two steps worth 1, a reset, then two steps worth 10.
    features = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [10.0, 0.0], [10.0, 0.0]]])
    dones = torch.tensor([[False, True, False, False]])

    buffer.add_rollout(features, dones)
    feats, mask, lengths = buffer.sample_episodes(batch_size=64, device="cpu")
    returns = buffer.discounted_feature_returns(feats, mask, gamma)[:, 0]

    # 1 + 0.5*1 = 1.5 and 10 + 0.5*10 = 15; never 1 + 0.5 + 0.25*10 + ...
    assert set(round(float(value), 4) for value in returns) == {1.5, 15.0}
    assert bool((lengths == 2).all())
