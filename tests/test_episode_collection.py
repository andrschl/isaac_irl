"""Complete episodes, real terminal successors, and honest vector-step costs."""

from dataclasses import replace

import pytest
import torch

from adapters.episode_collector import EpisodeCollector
from adapters.transition_collector import TransitionCollector
from interfaces import FeatureSpec
from storage.transition_replay import TransitionReplayBuffer


class _Tap:
    spec = FeatureSpec(1, ("success_bonus",))
    weights = torch.ones(1)
    step_dt = 0.1

    def features(self):
        return self.values

    def environment_reward(self):
        return self.values[:, 0] * self.step_dt

    def terminal_observations(self):
        return self.terminal


class _Env:
    device = torch.device("cpu")
    max_episode_length = 4

    def __init__(self, lengths=(2, 4, 3)):
        self.lengths = torch.tensor(lengths)
        self.num_envs = len(lengths)
        self.age = torch.zeros(self.num_envs, dtype=torch.int64)
        self.episode_length_buf = self.age
        self.tap = _Tap()
        self.resets = 0
        self.steps = 0
        self.ends_episodes = True

    def get_observations(self):
        return {
            "policy": torch.stack(
                (self.age.float(), torch.arange(self.num_envs).float()), 1
            )
        }

    def reset(self):
        self.resets += 1
        self.age.zero_()

    def step(self, actions):
        self.steps += 1
        self.age += 1
        dones = (self.age >= self.lengths) & self.ends_episodes
        timeouts = dones & (self.lengths == self.max_episode_length)
        self.tap.values = self.age.float()[:, None].clone()
        self.tap.terminal = self.get_observations()
        rewards = self.tap.environment_reward()
        self.age[dones] = 0
        return self.get_observations(), rewards, dones, {"time_outs": timeouts}


class _Policy:
    def __init__(self):
        self.model = torch.nn.Linear(2, 1)
        self.model.action_bound = 1.0

    def sample_with_log_probability(self, obs, *, generator=None):
        count = obs["policy"].shape[0]
        return torch.zeros(count, 1), torch.zeros(count)


def _collector(lengths=(2, 4, 3), episodes=5):
    env = _Env(lengths)
    steps = TransitionCollector(
        env, _Policy(), env.tap, rollout_steps=1, device="cpu", seed=0
    )
    return env, EpisodeCollector(steps, episodes_per_batch=episodes)


def test_multiple_waves_keep_complete_episodes_and_count_unused_lanes():
    env, collector = _collector()
    before = {
        name: value.clone()
        for name, value in collector.steps.policy.model.state_dict().items()
    }
    batch = collector.collect(policy_version=7)
    assert batch.mask.sum(1).tolist() == [2, 4, 3, 2, 4]
    assert batch.episode_count == 5
    assert batch.transition_count == 15
    assert batch.environment_transitions == 24
    assert batch.transitions.policy_version == 7
    assert env.resets == 2
    assert collector.maximum_transitions() == 24
    assert not batch.transitions.features[~batch.mask].any()
    torch.testing.assert_close(batch.returns, torch.tensor([0.3, 1.0, 0.6, 0.3, 1.0]))
    for name, value in collector.steps.policy.model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_replay_gets_only_valid_transitions_and_keeps_timeout_bootstrap():
    _, collector = _collector()
    batch = collector.collect(policy_version=0)
    flat = batch.valid_rollout()
    assert flat.transition_count == 15
    assert not flat.bootstrap_valid[flat.terminated].any()
    assert flat.bootstrap_valid[flat.truncated].all()
    assert (flat.next_critic_observations[flat.truncated][:, 0] == 4).all()
    replay = TransitionReplayBuffer(capacity_transitions=64)
    replay.add(flat)
    assert replay.transition_count == 15


def test_single_robot_sequential_episodes_have_no_lane_overhead():
    env, collector = _collector(lengths=(2,), episodes=10)
    batch = collector.collect(policy_version=0)
    assert batch.transition_count == batch.environment_transitions == 20
    assert env.resets == 10


def test_one_step_failure_is_a_complete_attempt_not_a_filtered_fragment():
    _, collector = _collector(lengths=(1,), episodes=3)
    batch = collector.collect(policy_version=0)
    assert batch.mask.sum(1).tolist() == [1, 1, 1]
    assert batch.transition_count == batch.environment_transitions == 3
    assert batch.valid_rollout().terminated.all()


def test_missing_episode_end_fails_instead_of_faking_a_terminal():
    env, collector = _collector()
    env.ends_episodes = False
    with pytest.raises(RuntimeError, match="explicit environment timeout"):
        collector.collect(policy_version=0)
    assert env.steps == env.max_episode_length


def test_invalidation_does_not_reset_the_environment():
    env, collector = _collector()
    collector.collect(policy_version=0)
    resets = env.resets
    collector.steps.invalidate_observations()
    assert env.resets == resets
    collector.collect(policy_version=1)
    assert env.resets == resets + 2


def test_episode_batch_rejects_nonprefix_mask_and_underreported_cost():
    _, collector = _collector()
    batch = collector.collect(policy_version=0)
    broken = batch.mask.clone()
    broken[1, 1] = False
    with pytest.raises(ValueError, match="contiguous"):
        replace(batch, mask=broken)
    with pytest.raises(ValueError, match="cannot be fewer"):
        replace(batch, environment_transitions=1)


def test_batch_ids_continue_across_checkpoint_reload_and_evaluation():
    _, collector = _collector()
    first = collector.collect(policy_version=0)
    state = collector.save_state()
    collector.load_state(state)
    second = collector.collect(
        policy_version=0, episodes=1, action_fn=lambda obs: torch.zeros(3, 1)
    )
    third = collector.collect(policy_version=1)
    assert [batch.transitions.rollout_id for batch in (first, second, third)] == [
        0,
        1,
        2,
    ]
