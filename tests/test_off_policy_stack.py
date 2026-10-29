# ruff: noqa: E402
"""Building and running the PMD stack, the core algorithm this repo is for."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import NUM_ENVS, OBS_DIM, FakeIsaacEnv, build_rollout  # noqa: E402

from adapters.off_policy_factory import build_off_policy_stack
from config.schema import (
    DirectQCfg,
    OffPolicyTrainCfg,
    PmdCfg,
    SuccessorQCfg,
    TransitionReplayCfg,
)
from storage.transition_replay import TransitionReplayBuffer


def _cfg(value) -> PmdCfg:
    return PmdCfg(
        train=OffPolicyTrainCfg(
            episodes_per_batch=4,
            critic_batch_size=8,
            critic_updates_per_batch=4,
            initial_critic_updates=0,
            replay=TransitionReplayCfg(learning_starts=1, capacity=1024),
        ),
        value=value,
        actor_batch_size=8,
        candidate_actions=4,
        projection_steps=2,
        policy_improvements_per_batch=1,
    )


@pytest.mark.parametrize("high_reuse", [False, True])
@pytest.mark.parametrize(
    "value",
    [
        DirectQCfg(num_heads=2, width=16, residual_blocks=1, target_subset_size=2),
        SuccessorQCfg(num_heads=2, width=16, residual_blocks=1, latent_dim=8),
    ],
    ids=["direct", "successor"],
)
def test_a_pmd_stack_collects_and_improves_on_the_environment_reward(
    value, high_reuse
) -> None:
    """Both value backends must survive a real collect/update cycle."""
    env = FakeIsaacEnv()
    cfg = _cfg(value)
    if high_reuse:
        cfg = replace(
            cfg,
            train=replace(
                cfg.train,
                critic_updates_per_batch=None,
                critic_updates_per_transition=2.0,
            ),
            policy_improvements_per_batch=4,
        )

    stack = build_off_policy_stack(
        env=env,
        observations=env.get_observations(),
        algorithm_cfg=cfg,
        ignored_reward_terms=set(),
        device="cpu",
        seed=0,
    )
    algorithm = stack.algorithm

    generation = algorithm.collect()
    metrics = algorithm.update()

    assert generation.transition_count == NUM_ENVS * 4
    # The features the reward learner would consume: one column per reward term.
    assert tuple(generation.features.shape) == (NUM_ENVS, 4, 2)
    assert metrics["Data/actor_updates"] >= 1.0
    assert "Value/q_mean" in metrics
    assert any(key.endswith("td_loss") for key in metrics)
    assert metrics["PMD/collection_policy_kl"] >= -1e-6
    if high_reuse:
        assert metrics["Data/critic_updates"] == 32
        assert metrics["Data/actor_updates"] == 4
        assert metrics["Data/critic_updates_per_transition"] == 2
        assert 4 <= metrics["Data/actor_optimizer_steps"] <= 8


def test_a_fixed_reward_stack_checks_that_features_reconstruct_the_reward() -> None:
    """The features and the reward must be the same quantity, or IRL is fitting noise."""
    env = FakeIsaacEnv()
    stack = build_off_policy_stack(
        env=env,
        observations=env.get_observations(),
        algorithm_cfg=_cfg(DirectQCfg(num_heads=2, width=16, residual_blocks=1)),
        ignored_reward_terms=set(),
        device="cpu",
        seed=0,
    )

    stack.algorithm.collect()
    metrics = stack.algorithm.update()

    assert metrics["Reward/reconstruction_max_error"] < 1.0e-5
    assert set(metrics) >= {"Reward/alive", "Reward/effort"}


def test_a_checkpoint_round_trips_through_the_stack() -> None:
    env = FakeIsaacEnv()
    stack = build_off_policy_stack(
        env=env,
        observations=env.get_observations(),
        algorithm_cfg=_cfg(SuccessorQCfg(num_heads=2, width=16, residual_blocks=1)),
        ignored_reward_terms=set(),
        device="cpu",
        seed=0,
    )
    stack.algorithm.collect()
    stack.algorithm.update()

    state = stack.algorithm.save_state()
    stack.algorithm.load_state(state, load_optimizer=True)

    assert stack.algorithm.learner.policy_version >= 1


def test_a_generation_reports_where_episodes_ended() -> None:
    """The reward learner needs episode boundaries, not just features.

    A termination and a time limit both end an episode for a discounted
    feature return, even though only a termination stops the Bellman backup.
    """
    env = FakeIsaacEnv()
    stack = build_off_policy_stack(
        env=env,
        observations=env.get_observations(),
        algorithm_cfg=_cfg(DirectQCfg(num_heads=2, width=16, residual_blocks=1)),
        ignored_reward_terms=set(),
        device="cpu",
        seed=0,
    )

    generation = stack.algorithm.collect()

    assert generation.dones is not None
    assert tuple(generation.dones.shape) == (NUM_ENVS, 4)
    assert generation.dones.dtype == torch.bool
    # Only the last step of each episode ends it, and it ends as a time limit.
    assert bool(generation.dones[:, -1].all())
    assert not bool(generation.dones[:, :-1].any())


def test_demo_initialization_trains_successor_and_entropy_without_stepping() -> None:
    env = FakeIsaacEnv(is_finite_horizon=True)
    cfg = _cfg(SuccessorQCfg(num_heads=2, width=16, residual_blocks=1, latent_dim=8))
    cfg = replace(
        cfg,
        entropy_rate=0.006,
        train=replace(cfg.train, gamma=1.0, initial_critic_updates=3),
    )
    algorithm = build_off_policy_stack(
        env=env,
        observations=env.get_observations(),
        algorithm_cfg=cfg,
        ignored_reward_terms=set(),
        device="cpu",
        seed=0,
    ).algorithm
    demos = build_rollout(num_envs=2, horizon=4, observation_dim=OBS_DIM)
    demos.truncated[:, -1] = True
    demos.bootstrap_valid[:, -1] = False
    algorithm.replay = TransitionReplayBuffer(
        capacity_transitions=100, demonstration_fraction=0.5
    )
    algorithm.replay.seed_with_demonstrations(demos)
    value = algorithm.learner.value
    value.fit_feature_scale(demos)
    modules = (value.online, value.target, value.entropy.online, value.entropy.target)
    before = [
        {name: tensor.clone() for name, tensor in module.state_dict().items()}
        for module in modules
    ]
    actor_before = {
        name: tensor.clone()
        for name, tensor in algorithm.actor.model.state_dict().items()
    }
    metrics = algorithm.warm_start_from_demonstrations()
    assert env.step_calls == 0
    assert algorithm.replay.transition_count == 0
    assert metrics["WarmStart/critic_updates"] == 3
    for module, initial in zip(modules, before, strict=True):
        assert any(
            not torch.equal(tensor, initial[name])
            for name, tensor in module.state_dict().items()
        )
    for policy in (algorithm.actor, algorithm.learner.previous_policy):
        for name, tensor in policy.model.state_dict().items():
            torch.testing.assert_close(tensor, actor_before[name], rtol=0, atol=0)


def test_a_finite_horizon_time_limit_stops_the_bootstrap() -> None:
    """On a task that really ends at H, the value past the horizon is zero.

    A continuing task truncated by a time limit still has a future worth
    bootstrapping. A finite-horizon one does not, and reading the flag off the
    environment is what keeps one collector correct for both.
    """
    for is_finite_horizon, bootstraps in ((False, True), (True, False)):
        env = FakeIsaacEnv(is_finite_horizon=is_finite_horizon)
        stack = build_off_policy_stack(
            env=env,
            observations=env.get_observations(),
            algorithm_cfg=_cfg(DirectQCfg(num_heads=2, width=16, residual_blocks=1)),
            ignored_reward_terms=set(),
            device="cpu",
            seed=0,
        )
        stack.algorithm.collect()
        rollout = stack.algorithm._pending_rollout
        assert bool(rollout.truncated[:, -1].all())
        assert not bool(rollout.terminated.any())
        assert bool(rollout.bootstrap_valid[:, -1].all()) is bootstraps


def test_evaluation_waits_for_update_and_preserves_training_state():
    env = FakeIsaacEnv()
    stack = build_off_policy_stack(
        env=env,
        observations=env.get_observations(),
        algorithm_cfg=_cfg(DirectQCfg(num_heads=2, width=16, residual_blocks=1)),
        ignored_reward_terms=set(),
        device="cpu",
        seed=0,
    )
    algorithm = stack.algorithm
    algorithm.collect()
    with pytest.raises(RuntimeError, match="pending training update"):
        algorithm.evaluate_episodes("stochastic", 4)
    algorithm.update()
    parameters = {
        name: value.clone()
        for name, value in algorithm.actor.model.state_dict().items()
    }
    version = algorithm.learner.policy_version
    replay_count = algorithm.replay.transition_count
    generator = algorithm.collector.steps._action_generator
    action_rng = generator.get_state().clone()
    torch_rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    reset = env.reset

    def random_reset():
        random.random()
        np.random.random()
        torch.rand(1)
        return reset()

    env.reset = random_reset
    batch = algorithm.evaluate_episodes("stochastic", 4)
    assert batch.episode_count == 4
    assert algorithm.replay.transition_count == replay_count
    assert algorithm.learner.policy_version == version
    assert torch.equal(generator.get_state(), action_rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert random.getstate() == python_rng
    actual_numpy_rng = np.random.get_state()
    np.testing.assert_array_equal(actual_numpy_rng[1], numpy_rng[1])
    assert actual_numpy_rng[2:] == numpy_rng[2:]
    for name, value in algorithm.actor.model.state_dict().items():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)


def _additive_reward(step_dt: float):
    """A fixed coefficient on 'alive' plus a learned one on 'effort'."""
    from config.schema import RewardModelCfg
    from reward.fixed import FixedLinearReward
    from reward.model import LinearFeatureRewardModel
    from reward.scorer import AdditiveLinearRewardScorer, LearnedLinearRewardScorer

    model = LinearFeatureRewardModel(RewardModelCfg(is_linear=True), num_features=1)
    with torch.no_grad():
        model.reward.weight.fill_(1.0)
    scorer = AdditiveLinearRewardScorer(
        FixedLinearReward(
            ("alive", "effort"), torch.tensor([2.0, 0.0]), step_dt=step_dt
        ),
        LearnedLinearRewardScorer(model, feature_names=("effort",), step_dt=step_dt),
    )
    return scorer, model


@pytest.mark.parametrize("coefficients", [(1.0, 1.0), (0.25, 0.75)])
def test_a_successor_stack_scalarizes_with_the_combined_reward(coefficients) -> None:
    """The fixed and learned halves meet at scalarization, not in the targets.

    Successor features carry occupancy only, so moving the learned reward has to
    move what the actor is scored against without moving what the critic
    predicts. That is the property that lets a reward keep changing during an
    off-policy run at all.
    """
    env = FakeIsaacEnv()
    scorer, model = _additive_reward(env.unwrapped.step_dt)
    scorer.fixed_coefficient, scorer.learned_coefficient = coefficients
    stack = build_off_policy_stack(
        env=env,
        observations=env.get_observations(),
        algorithm_cfg=_cfg(SuccessorQCfg(num_heads=2, width=16, residual_blocks=1)),
        ignored_reward_terms=set(),
        device="cpu",
        seed=0,
        reward=scorer,
    )
    stack.algorithm.collect()
    stack.algorithm.update()

    batch = stack.algorithm.replay.sample(
        8, generator=torch.Generator().manual_seed(0), device=torch.device("cpu")
    )
    value = stack.algorithm.learner.value
    with torch.no_grad():
        actions = stack.algorithm.actor.act_inference(batch.policy_observations)
        successors = value.online(batch.critic_observations, actions)
        before = value.mean(batch.critic_observations, actions, entropy_rate=0.0)
        model.reward.weight.fill_(-5.0)
        after = value.mean(batch.critic_observations, actions, entropy_rate=0.0)
        moved_successors = value.online(batch.critic_observations, actions)

    # The fixed coefficient is untouched by the reward learner ...
    torch.testing.assert_close(
        stack.reward.weights[0], torch.tensor(2.0 * coefficients[0])
    )
    torch.testing.assert_close(
        stack.reward.weights[1], torch.tensor(-5.0 * coefficients[1])
    )
    # ... the scalarized value follows the learned one ...
    assert not torch.allclose(before.reward_q, after.reward_q)
    # ... and the occupancy the critic predicts does not move at all.
    torch.testing.assert_close(moved_successors, successors, rtol=0, atol=0)


def test_a_cycle_steps_the_environment_once_per_transition_and_no_more() -> None:
    """Updates read replay. An update that stepped the simulator would be data.

    The batch is collected once under a frozen policy, inserted once, and then
    every optimizer step in the cycle draws from replay. Anything else would
    make the interaction count in the comparison a floor rather than a number.
    """
    env = FakeIsaacEnv()
    stack = build_off_policy_stack(
        env=env,
        observations=env.get_observations(),
        algorithm_cfg=_cfg(DirectQCfg(num_heads=2, width=16, residual_blocks=1)),
        ignored_reward_terms=set(),
        device="cpu",
        seed=0,
    )
    algorithm = stack.algorithm

    generation = algorithm.collect()
    steps_after_collection = env.step_calls
    replay_after_collection = algorithm.replay.transition_count
    algorithm.update()

    # One vector step per episode step; every lane is stepped in each of them.
    assert steps_after_collection == env.max_episode_length
    assert generation.transition_count == NUM_ENVS * env.max_episode_length
    assert replay_after_collection == generation.transition_count
    assert env.step_calls == steps_after_collection, "the update stepped the env"
    assert algorithm.replay.transition_count == replay_after_collection
