"""Reset events for the opt-in easier Franka Lift task."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def reset_franka_pregrasp(
    env: Any,
    env_ids: torch.Tensor | Sequence[int] | None,
    joint_positions: tuple[float, ...],
) -> None:
    """Reset selected robots to an open-gripper pose without changing defaults.

    joint_positions: seven arm angles in joint-name order, in radians.
    Writes joint position/velocity and actuator targets [B, J], where B is the
    selected environment count and J includes the gripper joints. Does not
    step physics, change action offsets, or consume any random numbers.
    """
    robot = env.scene["robot"]
    selected = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    )
    arm_ids, _ = robot.find_joints(
        [f"panda_joint{index}" for index in range(1, 8)], preserve_order=True
    )
    if len(arm_ids) != 7 or len(joint_positions) != 7:
        raise ValueError(
            "Pre-grasp reset requires seven named Franka arm joints and angles."
        )
    positions = robot.data.default_joint_pos[selected].clone()
    arm_positions = torch.as_tensor(
        joint_positions, dtype=positions.dtype, device=positions.device
    )
    if not torch.isfinite(arm_positions).all():
        raise ValueError("Pre-grasp joint angles must be finite.")
    positions[:, arm_ids] = arm_positions
    velocities = torch.zeros_like(positions)
    robot.write_joint_state_to_sim(positions, velocities, env_ids=selected)
    robot.set_joint_position_target(positions, env_ids=selected)
