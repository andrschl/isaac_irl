# ruff: noqa: E402
"""Does the critic actually estimate Q^pi?

Every other value-backend test checks one formula or that an update runs. This
one checks the outcome: on a chain whose discounted return is known in closed
form, the fitted critic has to converge to it. Rewards and dynamics here do not
depend on the action, so Q^pi(s,a) = V(s) for any policy and the analytic target
holds whatever the actor does - which is what makes a learned value comparable
to an exact one at all.

It exercises the whole estimation path together: reward scoring, the Bellman
target, bootstrapping through a Polyak target network, the per-head ensemble
split, terminal masking, and (for successor features) the output scale.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from algorithms.off_policy.actor import (
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
)
from algorithms.off_policy.values.backends import (
    DirectActionValue,
    SuccessorActionValue,
)
from config.schema import (
    DirectQCfg,
    OffPolicyTrainCfg,
    SuccessorQCfg,
    TransitionReplayCfg,
)
from reward.fixed import FixedLinearReward
from storage.transition_replay import TransitionReplayBuffer, TransitionRollout

NUM_STATES = 4
ACTION_DIM = 2
GAMMA = 0.9
REWARD_WEIGHTS = torch.tensor([1.0, 2.0, -1.0, 3.0])
FEATURE_NAMES = tuple(f"term_{index}" for index in range(NUM_STATES))


def analytic_action_values() -> torch.Tensor:
    """Exact Q^pi for the chain: the discounted sum of what is left. [S]."""
    values = torch.zeros(NUM_STATES)
    for state in reversed(range(NUM_STATES)):
        future = 0.0 if state == NUM_STATES - 1 else float(values[state + 1])
        values[state] = float(REWARD_WEIGHTS[state]) + GAMMA * future
    return values


def _one_hot(states: list[int]) -> torch.Tensor:
    return torch.eye(NUM_STATES)[states]


def _chain_rollout() -> TransitionRollout:
    """One transition out of every state: s -> s+1, with the last terminating.

    Shapes: observations [S,1,S], actions [S,1,A], features [S,1,S].
    """
    states = list(range(NUM_STATES))
    next_states = [min(state + 1, NUM_STATES - 1) for state in states]
    observations = _one_hot(states).unsqueeze(1)
    next_observations = _one_hot(next_states).unsqueeze(1)
    # The feature of a transition is the one-hot of the state it leaves, so the
    # reward of that step is REWARD_WEIGHTS[state].
    features = _one_hot(states).unsqueeze(1)
    terminated = torch.tensor([[state == NUM_STATES - 1] for state in states])
    return TransitionRollout(
        policy_observations=observations,
        next_policy_observations=next_observations,
        critic_observations=observations.clone(),
        next_critic_observations=next_observations.clone(),
        actions=torch.zeros(NUM_STATES, 1, ACTION_DIM),
        features=features,
        environment_rewards=(features.squeeze(1) @ REWARD_WEIGHTS).unsqueeze(1),
        terminated=terminated,
        truncated=torch.zeros_like(terminated),
        bootstrap_valid=~terminated,
        policy_version=0,
        rollout_id=0,
    )


def _replay() -> TransitionReplayBuffer:
    replay = TransitionReplayBuffer(capacity_transitions=64, device="cpu")
    replay.add(_chain_rollout())
    return replay


def _policy() -> StateDependentTanhPolicy:
    torch.manual_seed(0)
    return StateDependentTanhPolicy(
        StateDependentTanhGaussianActor(NUM_STATES, ACTION_DIM, hidden_dims=(16,))
    )


def _train_cfg() -> OffPolicyTrainCfg:
    return OffPolicyTrainCfg(
        gamma=GAMMA,
        critic_batch_size=64,
        critic_learning_rate=3.0e-3,
        # A fast target so a few thousand steps is enough to converge; the
        # default 0.005 measures the same thing far more slowly.
        target_polyak=0.05,
        replay=TransitionReplayCfg(learning_starts=1, capacity=64),
    )


def _fixed_reward() -> FixedLinearReward:
    return FixedLinearReward(FEATURE_NAMES, REWARD_WEIGHTS, step_dt=1.0)


def _learned_values(value, *, updates: int) -> torch.Tensor:
    """Run TD learning to convergence and score every state. [S]."""
    replay = _replay()
    policy = _policy()
    for _ in range(updates):
        value.update(replay, target_policy=policy, entropy_rate=0.0)
    with torch.no_grad():
        estimate = value.mean(
            _one_hot(list(range(NUM_STATES))),
            torch.zeros(NUM_STATES, ACTION_DIM),
            entropy_rate=0.0,
        )
    return estimate.q


def test_direct_q_converges_to_the_analytic_discounted_return() -> None:
    value = DirectActionValue(
        observation_dim=NUM_STATES,
        action_dim=ACTION_DIM,
        reward=_fixed_reward(),
        cfg=DirectQCfg(num_heads=2, width=64, residual_blocks=1, target_subset_size=2),
        train_cfg=_train_cfg(),
        step_dt=1.0,
        device=torch.device("cpu"),
        seed=0,
    )

    learned = _learned_values(value, updates=3000)

    torch.testing.assert_close(learned, analytic_action_values(), atol=0.25, rtol=0.05)


def test_successor_features_converge_to_the_analytic_feature_return() -> None:
    """Psi is the discounted feature occupancy; w^T Psi must be the same Q."""
    value = SuccessorActionValue(
        observation_dim=NUM_STATES,
        action_dim=ACTION_DIM,
        feature_dim=NUM_STATES,
        reward=_fixed_reward(),
        cfg=SuccessorQCfg(
            num_heads=2,
            width=64,
            residual_blocks=1,
            latent_dim=32,
            head_hidden_dim=32,
            # Score with the ensemble median: a lower quantile is deliberately
            # pessimistic, which is not what an accuracy check should measure.
            control_quantile=0.5,
        ),
        train_cfg=_train_cfg(),
        use_entropy_critic=False,
        step_dt=1.0,
        device=torch.device("cpu"),
        seed=0,
    )
    value.fit_feature_scale(_chain_rollout())

    learned = _learned_values(value, updates=3000)

    torch.testing.assert_close(learned, analytic_action_values(), atol=0.25, rtol=0.05)


def test_a_terminal_state_is_worth_only_its_own_reward() -> None:
    """If termination did not stop the backup, the last state would bootstrap."""
    analytic = analytic_action_values()

    assert float(analytic[-1]) == pytest.approx(float(REWARD_WEIGHTS[-1]))
    assert float(analytic[0]) == pytest.approx(
        float(
            REWARD_WEIGHTS[0]
            + GAMMA * REWARD_WEIGHTS[1]
            + GAMMA**2 * REWARD_WEIGHTS[2]
            + GAMMA**3 * REWARD_WEIGHTS[3]
        )
    )


def test_a_pessimistic_quantile_scores_below_the_median() -> None:
    """Pessimism has to cost something, or it is not doing anything."""

    def build(control_quantile: float) -> SuccessorActionValue:
        return SuccessorActionValue(
            observation_dim=NUM_STATES,
            action_dim=ACTION_DIM,
            feature_dim=NUM_STATES,
            reward=_fixed_reward(),
            cfg=SuccessorQCfg(
                num_heads=5,
                width=32,
                residual_blocks=1,
                latent_dim=16,
                head_hidden_dim=16,
                control_quantile=control_quantile,
            ),
            train_cfg=_train_cfg(),
            use_entropy_critic=False,
            step_dt=1.0,
            device=torch.device("cpu"),
            seed=0,
        )

    observations = _one_hot(list(range(NUM_STATES)))
    actions = torch.zeros(NUM_STATES, ACTION_DIM)
    median = build(0.5)
    pessimistic = build(0.0)
    median.fit_feature_scale(_chain_rollout())
    pessimistic.fit_feature_scale(_chain_rollout())
    # Same seed, so both ensembles hold identical heads: only scoring differs.
    pessimistic.online.load_state_dict(median.online.state_dict())

    with torch.no_grad():
        median_q = median.mean(observations, actions, entropy_rate=0.0).q
        pessimistic_q = pessimistic.mean(observations, actions, entropy_rate=0.0).q

    assert bool((pessimistic_q <= median_q + 1e-6).all())
