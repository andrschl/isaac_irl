"""Do staggered episode ages survive the things that reset the environment?

Environments that reset in lockstep deliver every post-reset reward to all of
them in the same iteration, so the replay alternates between blocks that carry
that reward and blocks that carry none. On Lift that is most of the reward -
the cube is still settling for the first steps of an episode, which fires the
lifting and goal-tracking terms for free - and the critic ends up fitting a
periodic artifact the policy cannot influence.

Randomizing episode ages fixes that only if the randomization outlives the
resets. It did not: the ages were spread before the collector's first lazy
`env.reset()`, which put every environment straight back to age zero, and
validation and the imitator rollout each undid it again later. These tests pin
all three paths.
"""

# ruff: noqa: E402

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from config.schema import FeatureBufCfg, RunnerCfg
from interfaces import FeatureSpec, Generation
from runner import Runner

NUM_ENVS = 16
MAX_EPISODE_LENGTH = 32


class _ResettingEnv:
    """Env stub whose reset zeroes episode ages, the way Isaac's does."""

    def __init__(self) -> None:
        self.num_envs = NUM_ENVS
        self.obs_dim = 4
        self.device = torch.device("cpu")
        self.max_episode_length = MAX_EPISODE_LENGTH
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.resets = 0

    def get_observations(self):
        return torch.zeros(self.num_envs, self.obs_dim)

    def reset(self):
        self.resets += 1
        self.episode_length_buf.zero_()
        return self.get_observations(), {}

    def step(self, actions: torch.Tensor):
        self.episode_length_buf += 1
        return (
            torch.zeros(self.num_envs, self.obs_dim),
            torch.ones(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {},
        )


class _Actor:
    def act(self, observations):
        return torch.zeros(NUM_ENVS, 2)

    def act_inference(self, observations):
        return torch.zeros(NUM_ENVS, 2)


class _RlAlg:
    """Algorithm stub that resets the env exactly where the real ones do."""

    def __init__(self, env: _ResettingEnv) -> None:
        self.env = env
        self.actor = _Actor()
        self.transition_count = NUM_ENVS * 4
        self._features = torch.ones(NUM_ENVS, 4, 3)
        self._dones = torch.zeros(NUM_ENVS, 4, dtype=torch.bool)
        self._observations = None

    @property
    def simulator_steps_per_update(self) -> int:
        return self.transition_count

    def collect(self) -> Generation:
        # The real collector resets lazily on its first call. That reset is
        # what used to wipe the stagger applied moments earlier in learn().
        if self._observations is None:
            self.reset_collection_state()
            self._observations = self.env.get_observations()
        self.env.step(None)
        return Generation(
            transition_count=self.transition_count,
            features=self._features,
            dones=self._dones,
        )

    def update(self) -> dict[str, float]:
        return {}

    def reset_collection_state(self) -> None:
        # The real adapters reset the environment here.
        self.env.reset()
        self._observations = self.env.get_observations()

    def train_metrics(self) -> dict[str, float]:
        return {}

    def eval_metrics(self) -> dict[str, float]:
        return {}

    def train_mode(self) -> None: ...

    def eval_mode(self) -> None: ...

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, checkpoint, *, load_optimizer: bool = True) -> None: ...


class _FeatureSource:
    def __init__(self, dim: int = 3) -> None:
        self.spec = FeatureSpec(dim=dim, names=tuple(f"term_{i}" for i in range(dim)))

    def features(self, env: Any) -> torch.Tensor:
        del env
        return torch.ones(NUM_ENVS, self.spec.dim)


def _runner(**cfg_overrides) -> tuple[Runner, _ResettingEnv]:
    env = _ResettingEnv()
    settings: dict[str, Any] = {
        "steps_per_env_per_cycle": 4,
        "validation_interval": 0,
        "imitator_buffer": FeatureBufCfg(min_ep_len=1),
    }
    settings.update(cfg_overrides)
    runner = Runner(
        env,
        rl_alg=_RlAlg(env),
        reward_learner=None,
        feature_source=_FeatureSource(),
        cfg=RunnerCfg(**settings),
        logger=None,
    )
    return runner, env


def _is_staggered(env: _ResettingEnv) -> bool:
    return bool((env.episode_length_buf != env.episode_length_buf[0]).any())


def test_staggering_survives_the_collectors_first_reset() -> None:
    """The regression: the stagger used to be applied and then reset away."""
    runner, env = _runner()
    torch.manual_seed(0)

    runner.learn(1, init_at_random_ep_len=True)

    assert _is_staggered(env), "every environment is still at the same episode age"


def test_ages_stay_spread_across_many_iterations() -> None:
    runner, env = _runner()
    torch.manual_seed(0)

    runner.learn(10, init_at_random_ep_len=True)

    assert _is_staggered(env)


def test_validation_restaggers_the_environments_it_reset() -> None:
    """Validation rolls out from a reset, so it has to put the spread back."""
    runner, env = _runner(validation_interval=1, validation_steps_per_env=2)
    torch.manual_seed(0)

    runner.learn(1, init_at_random_ep_len=True)

    assert env.resets > 1, "validation should have reset the environment"
    assert _is_staggered(env), "validation left every environment in lockstep"


def test_without_the_flag_the_ages_are_left_alone() -> None:
    """Staggering is opt-in; PPO-style runs must see an untouched env."""
    runner, env = _runner()
    torch.manual_seed(0)

    runner.learn(1, init_at_random_ep_len=False)

    assert not _is_staggered(env)
