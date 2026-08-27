from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adapters.rsl_rl_pre_reset_tap import IsaacPreResetTap  # noqa: E402
from reward_model.fixed import FixedLinearReward  # noqa: E402

NUM_ENVS = 4
STEP_DT = 0.05


class _FakeTermCfg:
    def __init__(self, weight: float, value: float) -> None:
        self.weight = weight
        self.value = value
        self.params: dict[str, object] = {}

    def func(self, env, **params):
        del params
        return torch.full((env.num_envs,), self.value)


class _FakeRewardManager:
    """Reward manager double with Isaac's per-step reward semantics."""

    def __init__(self, terms: list[tuple[str, float, float]]) -> None:
        self._term_names = [name for name, _, _ in terms]
        self._term_cfgs = [_FakeTermCfg(weight, value) for _, weight, value in terms]
        self._step_reward = torch.zeros(NUM_ENVS, len(terms))
        self.env: _FakeEnv | None = None

    def compute(self, dt: float) -> torch.Tensor:
        total = torch.zeros(NUM_ENVS)
        for index, cfg in enumerate(self._term_cfgs):
            if cfg.weight == 0.0:
                # Isaac skips zero-weight terms entirely.
                self._step_reward[:, index] = 0.0
                continue
            weighted = cfg.func(self.env) * cfg.weight
            self._step_reward[:, index] = weighted
            total += weighted * dt
        return total


class _FakeTerminationManager:
    time_outs = torch.zeros(NUM_ENVS, dtype=torch.bool)


class _FakeObservationManager:
    def compute(self):
        return {"policy": torch.zeros(NUM_ENVS, 2)}


class _FakeEnv:
    def __init__(self, terms: list[tuple[str, float, float]]) -> None:
        self.num_envs = NUM_ENVS
        self.device = "cpu"
        self.step_dt = STEP_DT
        self.reward_manager = _FakeRewardManager(terms)
        self.reward_manager.env = self
        self.termination_manager = _FakeTerminationManager()
        self.observation_manager = _FakeObservationManager()

    @property
    def unwrapped(self):
        return self


TERMS = [
    ("alive", 1.0, 1.0),
    ("pole_pos", -1.0, 0.25),
    ("disabled", 0.0, 0.5),
    ("success_bonus", 0.0, 1.0),
]


def _tapped(**kwargs) -> tuple[_FakeEnv, IsaacPreResetTap]:
    env = _FakeEnv(TERMS)
    tap = IsaacPreResetTap(env, device="cpu", **kwargs)
    tap.install()
    env.reward_manager.compute(STEP_DT)
    return env, tap


def test_zero_weight_terms_are_dropped_unless_requested() -> None:
    _, tap = _tapped()

    assert tap.spec.names == ("alive", "pole_pos")
    torch.testing.assert_close(tap.weights, torch.tensor([1.0, -1.0]))


def test_feature_only_terms_keep_reward_manager_order() -> None:
    _, tap = _tapped(force_include_terms={"success_bonus"})

    assert tap.spec.names == ("alive", "pole_pos", "success_bonus")
    torch.testing.assert_close(tap.weights, torch.tensor([1.0, -1.0, 0.0]))


def test_feature_only_terms_are_captured_at_their_raw_value() -> None:
    _, tap = _tapped(force_include_terms={"success_bonus"})

    features = tap.features()

    # Raw unweighted values, including the term the manager never evaluates.
    torch.testing.assert_close(
        features, torch.tensor([[1.0, 0.25, 1.0]]).repeat(NUM_ENVS, 1)
    )


def test_captured_features_reconstruct_the_environment_reward() -> None:
    _, tap = _tapped(force_include_terms={"success_bonus"})
    reward = FixedLinearReward(tap.spec.names or (), tap.weights, step_dt=tap.step_dt)

    features = tap.features()
    error = reward.reconstruction_error(features, tap.environment_reward())

    assert float(error.max().item()) < 1.0e-6


def test_ignored_terms_are_excluded() -> None:
    _, tap = _tapped(ignored_terms={"pole_pos"})

    assert tap.spec.names == ("alive",)


def test_a_reward_manager_without_captured_terms_is_rejected() -> None:
    env = _FakeEnv([("disabled", 0.0, 1.0)])

    with pytest.raises(ValueError, match="no captured terms"):
        IsaacPreResetTap(env, device="cpu")
