from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from reward_features.manager_based import (
    ManagerBasedFeatureCfg,
    ManagerBasedRewardFeatureEncoder,
    _as_term_vector,
    manager_based_reward_features,
)


@dataclass(slots=True)
class _TermCfg:
    weight: float
    params: dict | None
    func: callable


class _RewardManager:
    def __init__(self, term_names, term_cfgs) -> None:
        self._term_names = term_names
        self._term_cfgs = term_cfgs


class _Env:
    def __init__(self, num_envs: int, reward_manager: _RewardManager) -> None:
        self.unwrapped = type(
            "_UnwrappedEnv",
            (),
            {"num_envs": int(num_envs), "reward_manager": reward_manager},
        )()


def test_as_term_vector_supports_scalar_1d_and_n1():
    scalar_out = _as_term_vector(torch.tensor(2.0), num_envs=3)
    assert tuple(scalar_out.shape) == (3,)
    assert torch.allclose(scalar_out, torch.tensor([2.0, 2.0, 2.0]))

    one_d_out = _as_term_vector(torch.tensor([1.0, 3.0, 5.0]), num_envs=3)
    assert tuple(one_d_out.shape) == (3,)
    assert torch.allclose(one_d_out, torch.tensor([1.0, 3.0, 5.0]))

    n1_out = _as_term_vector(torch.tensor([[4.0], [6.0], [8.0]]), num_envs=3)
    assert tuple(n1_out.shape) == (3,)
    assert torch.allclose(n1_out, torch.tensor([4.0, 6.0, 8.0]))


def test_as_term_vector_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="expected first dim 3"):
        _as_term_vector(torch.tensor([1.0, 2.0]), num_envs=3)

    with pytest.raises(ValueError, match="Unsupported reward term output shape"):
        _as_term_vector(torch.ones(3, 2), num_envs=3)


def test_manager_based_reward_features_filters_zero_weight_and_ignored():
    num_envs = 2

    def _term_a(env):
        del env
        return torch.tensor([1.0, 2.0])

    def _term_b(env):
        del env
        return torch.tensor([[10.0], [20.0]])

    def _term_ignored(env):
        del env
        return torch.tensor([99.0, 99.0])

    reward_manager = _RewardManager(
        term_names=["a", "b", "ignored", "zero_weight"],
        term_cfgs=[
            _TermCfg(weight=1.0, params=None, func=_term_a),
            _TermCfg(weight=1.0, params={}, func=_term_b),
            _TermCfg(weight=1.0, params=None, func=_term_ignored),
            _TermCfg(weight=0.0, params=None, func=lambda env: torch.tensor([7.0, 7.0])),
        ],
    )
    env = _Env(num_envs=num_envs, reward_manager=reward_manager)

    feats = manager_based_reward_features(env=env, ignored_reward_terms={"ignored"})
    assert tuple(feats.shape) == (2, 2)
    assert torch.allclose(
        feats,
        torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
            ]
        ),
    )


def test_manager_based_reward_features_returns_empty_feature_matrix_when_all_filtered():
    reward_manager = _RewardManager(
        term_names=["only_zero"],
        term_cfgs=[_TermCfg(weight=0.0, params=None, func=lambda env: torch.tensor([1.0]))],
    )
    env = _Env(num_envs=1, reward_manager=reward_manager)
    feats = manager_based_reward_features(env=env)
    assert tuple(feats.shape) == (1, 0)
    assert feats.dtype == torch.float32


def test_manager_based_feature_encoder_calls_manager_features():
    reward_manager = _RewardManager(
        term_names=["t"],
        term_cfgs=[_TermCfg(weight=1.0, params=None, func=lambda env: torch.tensor([3.0, 4.0]))],
    )
    env = _Env(num_envs=2, reward_manager=reward_manager)
    encoder = ManagerBasedRewardFeatureEncoder(
        env=env,
        cfg=ManagerBasedFeatureCfg(ignored_reward_terms=set()),
        device="cpu",
    )
    out = encoder()
    assert tuple(out.shape) == (2, 1)
    assert torch.allclose(out[:, 0], torch.tensor([3.0, 4.0]))
