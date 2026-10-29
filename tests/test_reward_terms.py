# ruff: noqa: E402
"""Selecting reward-manager terms, and reading them as features."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from config.schema import FeatureMapCfg
from reward.features import ManagerBasedFeatureSource
from reward.manager_terms import as_term_vector, select_reward_terms


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


def _constant(*values: float):
    return lambda env: torch.tensor(list(values))


def _manager(**terms: _TermCfg) -> _RewardManager:
    return _RewardManager(list(terms), list(terms.values()))


def test_a_term_value_is_normalized_to_one_number_per_env():
    assert torch.allclose(
        as_term_vector(torch.tensor(2.0), num_envs=3), torch.tensor([2.0, 2.0, 2.0])
    )
    assert torch.allclose(
        as_term_vector(torch.tensor([1.0, 3.0, 5.0]), num_envs=3),
        torch.tensor([1.0, 3.0, 5.0]),
    )
    assert torch.allclose(
        as_term_vector(torch.tensor([[4.0], [6.0], [8.0]]), num_envs=3),
        torch.tensor([4.0, 6.0, 8.0]),
    )


@pytest.mark.parametrize(
    "value", [torch.tensor([1.0, 2.0]), torch.ones(3, 2), torch.ones(2, 3)]
)
def test_a_term_value_of_the_wrong_shape_is_refused(value) -> None:
    with pytest.raises(ValueError, match="expected a scalar"):
        as_term_vector(value, num_envs=3)


def test_zero_weight_and_ignored_terms_are_left_out():
    manager = _manager(
        reach=_TermCfg(1.0, None, _constant(1.0, 2.0)),
        lift=_TermCfg(2.0, {}, _constant(10.0, 20.0)),
        unwanted=_TermCfg(1.0, None, _constant(99.0, 99.0)),
        disabled=_TermCfg(0.0, None, _constant(7.0, 7.0)),
    )

    terms = select_reward_terms(manager, ignored_terms={"unwanted"})

    assert [term.name for term in terms] == ["reach", "lift"]
    assert [term.index for term in terms] == [0, 1], (
        "manager order is the feature order"
    )


def test_a_zero_weight_term_becomes_a_feature_only_when_asked_for():
    """A success bonus injected at weight 0 is a feature, not a reward term."""
    manager = _manager(
        reach=_TermCfg(1.0, None, _constant(1.0)),
        success_bonus=_TermCfg(0.0, None, _constant(1.0)),
    )

    without = select_reward_terms(manager)
    with_bonus = select_reward_terms(manager, force_include_terms={"success_bonus"})

    assert [term.name for term in without] == ["reach"]
    assert [term.name for term in with_bonus] == ["reach", "success_bonus"]
    assert with_bonus[1].contributes_to_reward is False


def test_selecting_nothing_is_an_error_rather_than_an_empty_feature_space():
    """An empty feature vector is a misconfiguration, not a valid reward space."""
    manager = _manager(only_zero=_TermCfg(0.0, None, _constant(1.0)))

    with pytest.raises(ValueError, match="no feature terms"):
        select_reward_terms(manager)


def test_the_feature_source_fixes_names_without_evaluating_any_term():
    """Building a feature source must not run reward terms against the env."""
    evaluations: list[str] = []

    def _recording(env):
        del env
        evaluations.append("called")
        return torch.tensor([3.0, 4.0])

    env = _Env(2, _manager(t=_TermCfg(1.0, None, _recording)))

    source = ManagerBasedFeatureSource(env=env, cfg=FeatureMapCfg(), device="cpu")

    assert source.spec.dim == 1 and source.spec.names == ("t",)
    assert evaluations == []

    features = source.features(env)

    assert tuple(features.shape) == (2, 1)
    assert torch.allclose(features[:, 0], torch.tensor([3.0, 4.0]))
    assert evaluations == ["called"]


def test_the_tap_and_the_feature_source_select_the_same_terms():
    """Demonstrations and online features must describe one reward space."""
    from adapters.pre_reset_tap import IsaacPreResetTap

    manager = _manager(
        reach=_TermCfg(1.0, None, _constant(1.0, 1.0)),
        disabled=_TermCfg(0.0, None, _constant(0.0, 0.0)),
        success_bonus=_TermCfg(0.0, None, _constant(1.0, 0.0)),
    )
    env = _Env(2, manager)
    env.device = "cpu"

    source = ManagerBasedFeatureSource(
        env=env, cfg=FeatureMapCfg(force_include_terms={"success_bonus"})
    )
    tap = IsaacPreResetTap(env, force_include_terms={"success_bonus"}, device="cpu")

    assert source.spec == tap.spec
