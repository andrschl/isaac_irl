"""Named task geometry presets, applied before constructing the Isaac env."""

from __future__ import annotations

from typing import Any

from adapters.lift_reset import reset_franka_pregrasp

# Calibrated in Isaac with the existing Franka asset and joint-position action.
# Downward open gripper: fingertip-center frame near (0.499, 0, 0.092) m.
# These are reset angles only; default_joint_pos and action offsets stay native.
FRANKA_PREGRASP_JOINT_POSITIONS = (
    -0.0598707981,
    0.2905777991,
    0.0555725731,
    -2.3991672993,
    -0.0361596346,
    2.6847262383,
    0.8107280731,
)


def configure_lift_easy(env_cfg: Any, *, cube_xy_half_range: float = 0.02) -> None:
    """Narrow cube resets, set a nearby goal, and start above the ungrasped cube.

    Coordinates are metres in the native task's robot/environment frame.
    Reward terms, termination rules, horizon, dynamics, and actions are untouched.
    """
    # Measured resting center height: avoid native 5.5-cm airborne reset credit.
    env_cfg.scene.object.init_state.pos = (0.5, 0.0, 0.021)
    env_cfg.events.reset_object_position.params["pose_range"] = {
        "x": (-cube_xy_half_range, cube_xy_half_range),
        "y": (-cube_xy_half_range, cube_xy_half_range),
        "z": (0.0, 0.0),
    }
    ranges = env_cfg.commands.object_pose.ranges
    ranges.pos_x = (0.5, 0.5)
    ranges.pos_y = (0.0, 0.0)
    ranges.pos_z = (0.2, 0.2)
    # Append after the native reset events; preserve their reset ordering.
    env_cfg.events.reset_pregrasp = env_cfg.events.reset_all.replace(
        func=reset_franka_pregrasp,
        params={"joint_positions": FRANKA_PREGRASP_JOINT_POSITIONS},
    )
