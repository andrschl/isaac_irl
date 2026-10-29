"""The sparse finite-horizon task: its state, its feature basis, its reward."""

# ruff: noqa: E402

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from config.schema import RewardModelCfg
from reward.features import FeatureColumns
from reward.fixed import FixedLinearReward
from reward.model import LinearFeatureRewardModel
from reward.scorer import AdditiveLinearRewardScorer, LearnedLinearRewardScorer
from reward.terminal_success import TerminalSuccessState

HORIZON = 6
HOLD = 3


def _state(num_envs: int = 2, *, hold: int = HOLD) -> TerminalSuccessState:
    return TerminalSuccessState(
        num_envs, horizon=HORIZON, hold_steps=hold, device="cpu"
    )


def _flags(values: list[bool]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


def _run(
    state: TerminalSuccessState,
    reached_per_step: list[list[bool]],
    *,
    failed_at: int | None = None,
    num_envs: int = 2,
) -> torch.Tensor:
    """Drive one episode and return the success pulse of each step. [T, N]."""
    pulses = []
    for index, reached in enumerate(reached_per_step):
        age = index + 1
        failed = _flags([failed_at == age] * num_envs)
        state.update(
            _flags(reached),
            failed,
            torch.full((num_envs,), age, dtype=torch.long),
            step=age,
        )
        pulses.append(state.success.clone())
    return torch.stack(pulses)


def test_success_is_paid_once_for_a_goal_held_to_the_horizon() -> None:
    """Reaching early pays nothing; only the last transition can pay."""
    state = _state()
    reached = [[False, True]] * (HORIZON - HOLD) + [[True, True]] * HOLD

    pulses = _run(state, reached)

    # Env 0 reached only for the final HOLD steps; env 1 held throughout.
    assert bool(pulses[-1].all())
    assert not bool(pulses[:-1].any()), "no step before the horizon pays"


def test_letting_go_on_the_final_step_forfeits_the_whole_episode() -> None:
    state = _state(num_envs=1)

    pulses = _run(state, [[True]] * (HORIZON - 1) + [[False]], num_envs=1)

    assert not bool(pulses.any())


def test_an_interrupted_hold_has_to_be_earned_again() -> None:
    """The streak is consecutive, so a lapse inside the window is not forgiven."""
    state = _state(num_envs=1)
    # Reached everywhere except two steps before the end: the final run is 2 < 3.
    reached = [[True]] * HORIZON
    reached[HORIZON - HOLD] = [False]

    pulses = _run(state, reached, num_envs=1)

    assert not bool(pulses.any())
    assert int(state.streak[0]) == HOLD - 1


def test_a_failure_on_the_final_step_beats_the_goal() -> None:
    state = _state(num_envs=1)

    pulses = _run(state, [[True]] * HORIZON, failed_at=HORIZON, num_envs=1)

    assert not bool(pulses.any())


def test_a_one_step_failure_leaves_nothing_to_pay() -> None:
    """An episode that dies at its first transition is still a real episode."""
    state = _state(num_envs=1)

    state.update(
        _flags([True]), _flags([True]), torch.ones(1, dtype=torch.long), step=1
    )

    assert not bool(state.success.any())
    assert int(state.streak[0]) == 1


def test_evaluating_a_step_twice_does_not_advance_it_twice() -> None:
    """The observation manager computes every term twice per physical step."""
    state = _state(num_envs=1)
    ages = torch.ones(1, dtype=torch.long)

    state.update(_flags([True]), _flags([False]), ages, step=1)
    state.update(_flags([True]), _flags([False]), ages, step=1)

    assert int(state.streak[0]) == 1


def test_reading_the_episode_context_never_advances_the_streak() -> None:
    state = _state(num_envs=1)
    state.update(
        _flags([True]), _flags([False]), torch.ones(1, dtype=torch.long), step=1
    )

    first = state.observations(torch.ones(1, dtype=torch.long))
    second = state.observations(torch.ones(1, dtype=torch.long))

    torch.testing.assert_close(first, second)
    assert tuple(first.shape) == (1, 2)
    # Remaining time, then hold progress, both in [0, 1].
    torch.testing.assert_close(
        first[0], torch.tensor([1.0 - 1.0 / HORIZON, 1.0 / HOLD])
    )
    assert int(state.streak[0]) == 1


def test_one_lane_resets_without_disturbing_the_others() -> None:
    """Isaac recycles lanes independently, and so must the state they share."""
    state = _state(num_envs=3)
    ages = torch.ones(3, dtype=torch.long)
    state.update(_flags([True] * 3), _flags([False] * 3), ages, step=1)

    state.reset(torch.tensor([1]))

    assert state.streak.tolist() == [1, 0, 1]
    assert state.last_step.tolist() == [1, -1, 1]
    # The reset lane starts its own episode at the next physical step.
    state.update(_flags([True] * 3), _flags([False] * 3), torch.full((3,), 2), step=2)
    assert state.streak.tolist() == [2, 1, 2]


def test_a_hold_longer_than_the_episode_is_rejected() -> None:
    with pytest.raises(ValueError, match="hold_steps <= horizon"):
        TerminalSuccessState(2, horizon=4, hold_steps=5, device="cpu")


# ----------------------------------------------------------- feature basis


def test_the_reward_learner_basis_drops_the_sparse_coordinate() -> None:
    names = ("reaching", "lifting", "terminal_success")
    columns = FeatureColumns.select(names, names[:-1])

    features = torch.tensor([[[1.0, 2.0, 1.0]]])

    assert columns.dim == 2
    assert not columns.is_identity
    torch.testing.assert_close(columns.apply(features), torch.tensor([[[1.0, 2.0]]]))


def test_an_identity_selection_hands_the_features_straight_back() -> None:
    columns = FeatureColumns.identity(("a", "b"))
    features = torch.ones(2, 3, 2)

    assert columns.is_identity
    assert columns.apply(features) is features


def test_asking_for_a_feature_the_environment_does_not_report_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"\['spin'\] are not collected"):
        FeatureColumns.select(("a", "b"), ("a", "spin"))


# ------------------------------------------------------------ additive reward


def _additive(
    names: tuple[str, ...], learned_names: tuple[str, ...], *, step_dt: float = 0.02
) -> tuple[AdditiveLinearRewardScorer, LinearFeatureRewardModel]:
    model = LinearFeatureRewardModel(
        RewardModelCfg(is_linear=True), num_features=len(learned_names)
    )
    fixed_weights = torch.tensor(
        [1.0 if name == "terminal_success" else 0.0 for name in names]
    )
    scorer = AdditiveLinearRewardScorer(
        FixedLinearReward(names, fixed_weights, step_dt=step_dt),
        LearnedLinearRewardScorer(model, feature_names=learned_names, step_dt=step_dt),
    )
    return scorer, model


def test_the_sparse_coefficient_survives_every_reward_update() -> None:
    """The task's own coefficient is fixed; the reward learner cannot move it."""
    names = ("reaching", "terminal_success")
    scorer, model = _additive(names, ("reaching",))

    with torch.no_grad():
        model.reward.weight.fill_(-3.0)

    torch.testing.assert_close(scorer.weights, torch.tensor([-3.0, 1.0]))


def test_a_reward_update_is_visible_to_the_very_next_scoring_call() -> None:
    names = ("reaching", "terminal_success")
    scorer, model = _additive(names, ("reaching",))
    features = torch.tensor([[2.0, 0.0]])

    with torch.no_grad():
        model.reward.weight.fill_(0.5)
    before = scorer.score(features)
    with torch.no_grad():
        model.reward.weight.fill_(1.5)
    after = scorer.score(features)

    assert float(before) == pytest.approx(0.02 * 0.5 * 2.0)
    assert float(after) == pytest.approx(0.02 * 1.5 * 2.0)


def test_the_terminal_reward_integrates_to_exactly_one_unit() -> None:
    """Isaac pays ``func * weight * dt``, so the raw feature is ``g / dt``.

    The whole sparse arm is defined by that single unit arriving once, so the
    round trip through the tap's raw-feature convention has to be exact rather
    than approximately right.
    """
    step_dt = 0.02
    names = ("reaching", "terminal_success")
    scorer, model = _additive(names, ("reaching",), step_dt=step_dt)
    with torch.no_grad():
        model.reward.weight.zero_()

    # What the pre-reset tap records on the successful final transition.
    features = torch.tensor([[0.0, 1.0 / step_dt]])

    assert float(scorer.score(features)) == pytest.approx(1.0)


def test_a_learned_feature_outside_the_shared_basis_is_refused() -> None:
    model = LinearFeatureRewardModel(RewardModelCfg(is_linear=True), num_features=1)
    with pytest.raises(ValueError, match="not in the fixed reward basis"):
        AdditiveLinearRewardScorer(
            FixedLinearReward(("a",), torch.zeros(1), step_dt=0.02),
            LearnedLinearRewardScorer(model, feature_names=("b",), step_dt=0.02),
        )


def test_the_two_halves_must_agree_on_the_environment_step() -> None:
    model = LinearFeatureRewardModel(RewardModelCfg(is_linear=True), num_features=1)
    with pytest.raises(ValueError, match="same step_dt"):
        AdditiveLinearRewardScorer(
            FixedLinearReward(("a",), torch.zeros(1), step_dt=0.02),
            LearnedLinearRewardScorer(model, feature_names=("a",), step_dt=0.05),
        )
