"""Isaac binding for the finite-horizon terminal-success task on Franka Lift.

Import only after ``AppLauncher`` has started the simulator: it pulls in
``isaaclab``. One :class:`TerminalSuccessState` is attached to the environment
and shared by two terms with different jobs - the reward term advances it once
per physical step, the observation term only reads it.
"""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab_tasks.manager_based.manipulation.lift.mdp.terminations import (
    object_reached_goal,
)

from reward.terminal_success import TERMINAL_SUCCESS_TERM, TerminalSuccessState

STATE_ATTRIBUTE = "_terminal_success_state"
REWARD_TERM_NAME = TERMINAL_SUCCESS_TERM


def terminal_state(env: Any) -> TerminalSuccessState:
    """Return the environment's shared terminal-success state, building it once.

    The observation manager evaluates its terms while it is being constructed,
    which happens before the reward manager exists, so whichever term runs
    first has to be able to build the state. Both read the same reward-term
    configuration to do it, so there is one source for the horizon and hold.
    """
    state = getattr(env, STATE_ATTRIBUTE, None)
    if state is not None:
        return state
    cfg = getattr(env.cfg.rewards, REWARD_TERM_NAME, None)
    if cfg is None:
        raise ValueError(
            "Terminal-success terms need a "
            f"`rewards.{REWARD_TERM_NAME}` term on the environment configuration."
        )
    state = TerminalSuccessState(
        int(env.num_envs),
        horizon=int(env.max_episode_length),
        hold_steps=int(cfg.params["hold_steps"]),
        device=env.device,
    )
    setattr(env, STATE_ATTRIBUTE, state)
    return state


def episode_context(env: Any) -> torch.Tensor:
    """Student observation tail [N, 2]; never advances the task state."""
    return terminal_state(env).observations(env.episode_length_buf)


class LiftTerminalSuccess(ManagerTermBase):
    """Pay one, once, for a goal held to the horizon.

    Isaac integrates reward terms as ``func * weight * dt``, so this returns
    ``g / dt`` and a weight of one pays exactly ``g`` on the final transition.
    """

    def __call__(self, env: Any, threshold: float, hold_steps: int) -> torch.Tensor:
        state = terminal_state(env)
        if hold_steps != state.hold_steps:
            raise ValueError(
                f"Terminal-success hold changed from {state.hold_steps} to "
                f"{hold_steps} after the state was built."
            )
        state.update(
            object_reached_goal(env, command_name="object_pose", threshold=threshold),
            env.termination_manager.terminated,
            env.episode_length_buf,
            step=int(env.common_step_counter),
        )
        return state.success.to(torch.float32) / env.step_dt

    def reset(self, env_ids: Any = None) -> None:
        """Isaac's reward manager owns the reset; the observation term must not."""
        terminal_state(self._env).reset(slice(None) if env_ids is None else env_ids)
