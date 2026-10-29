# ruff: noqa: E402
"""Does the VecEnv adapter proxy in both directions?

The adapter exists to rename one field. Everything else has to reach the
wrapped environment untouched - including assignments. A read-only proxy looks
correct in every test that only reads: the write lands on the adapter, shadows
the environment's attribute, and the next read returns the value that was just
written. The environment never sees it.

That is not hypothetical. Episode-age staggering, which desynchronizes resets
so post-reset reward does not arrive in every environment at once, was written
through this adapter and silently dropped for exactly this reason.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from adapters.vec_env import RslRlVecEnvAdapter

NUM_ENVS = 4
MAX_EPISODE_LENGTH = 200


class _RecordingEnv:
    """The environment the adapter wraps, with one settable buffer."""

    num_actions = 8
    num_envs = NUM_ENVS
    max_episode_length = MAX_EPISODE_LENGTH

    def __init__(self) -> None:
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)


def test_action_dim_renames_num_actions() -> None:
    env = _RecordingEnv()
    adapter = RslRlVecEnvAdapter(env)

    assert adapter.action_dim == env.num_actions
    # The legacy name still proxies, so rsl_rl code paths keep working.
    assert adapter.num_actions == env.num_actions


def test_reads_proxy_to_the_wrapped_environment() -> None:
    adapter = RslRlVecEnvAdapter(_RecordingEnv())

    assert adapter.num_envs == NUM_ENVS
    assert adapter.max_episode_length == MAX_EPISODE_LENGTH


def test_a_write_through_the_adapter_reaches_the_environment() -> None:
    """The regression: staggered episode ages must not stop at the wrapper."""
    env = _RecordingEnv()
    adapter = RslRlVecEnvAdapter(env)

    staggered = torch.arange(NUM_ENVS, dtype=torch.long)
    adapter.episode_length_buf = staggered

    torch.testing.assert_close(env.episode_length_buf, staggered)
    # And the adapter does not hold a shadowing copy of its own.
    assert "episode_length_buf" not in vars(adapter)


def test_the_adapter_keeps_its_own_reference_to_the_environment() -> None:
    """`_env` is the one name that must not be proxied, or __init__ recurses."""
    env = _RecordingEnv()
    adapter = RslRlVecEnvAdapter(env)

    assert vars(adapter)["_env"] is env
    assert not hasattr(env, "_env")
