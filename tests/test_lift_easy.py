"""The easy preset changes initial/goal distributions, not learning semantics."""

import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from adapters.lift_reset import reset_franka_pregrasp
from config import load_experiment_cfg
from config.schema import EnvCfg, RewardCfg, TrainCfg
from training.build import build_env_cfg
from training.env_variants import FRANKA_PREGRASP_JOINT_POSITIONS


class _Event(SimpleNamespace):
    def replace(self, **changes):
        result = deepcopy(self)
        vars(result).update(changes)
        return result


def _native_config():
    return SimpleNamespace(
        scene=SimpleNamespace(
            robot=SimpleNamespace(init_state=SimpleNamespace(joint_pos={"native": 1})),
            object=SimpleNamespace(init_state=SimpleNamespace(pos=(0.5, 0, 0.055))),
        ),
        commands=SimpleNamespace(
            object_pose=SimpleNamespace(
                ranges=SimpleNamespace(
                    pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5)
                )
            )
        ),
        events=SimpleNamespace(
            reset_all=_Event(func=lambda: None, mode="reset", params={}),
            reset_object_position=_Event(params={"pose_range": {"x": (-0.1, 0.1)}}),
        ),
        rewards=object(),
        actions=object(),
        terminations=object(),
        episode_length_s=5,
    )


@pytest.mark.parametrize("variant", ["standard", "lift_easy"])
def test_easy_config_preserves_native_control_and_reward(variant):
    native = _native_config()
    preserved = (
        native.rewards,
        native.actions,
        native.terminations,
        native.scene.robot,
    )
    cfg = TrainCfg(
        env=EnvCfg(
            name="Isaac-Lift-Cube-Franka-v0", variant=variant, add_success_bonus=False
        ),
        reward=RewardCfg(source="environment"),
    )
    result = build_env_cfg(
        cfg, parse_env_cfg=lambda *args, **kwargs: native, use_fabric=True
    )
    assert (
        result.rewards,
        result.actions,
        result.terminations,
        result.scene.robot,
    ) == preserved
    assert result.episode_length_s == 5
    if variant == "standard":
        assert result.scene.object.init_state.pos == (0.5, 0, 0.055)
        assert not hasattr(result.events, "reset_pregrasp")
    else:
        assert result.scene.object.init_state.pos == (0.5, 0, 0.021)
        assert result.commands.object_pose.ranges.pos_z == (0.2, 0.2)
        assert result.events.reset_object_position.params["pose_range"]["y"] == (
            -0.02,
            0.02,
        )
        assert result.events.reset_pregrasp.func is reset_franka_pregrasp
        assert result.events.reset_all.params == {}


class _Robot:
    def __init__(self):
        self.data = SimpleNamespace(default_joint_pos=torch.full((4, 9), 0.04))
        self.positions = torch.zeros(4, 9)
        self.velocities = torch.ones(4, 9)
        self.targets = torch.zeros(4, 9)

    def find_joints(self, names, *, preserve_order):
        assert names == [f"panda_joint{index}" for index in range(1, 8)]
        assert preserve_order
        return list(range(7)), names

    def write_joint_state_to_sim(self, positions, velocities, *, env_ids):
        self.positions[env_ids] = positions
        self.velocities[env_ids] = velocities

    def set_joint_position_target(self, positions, *, env_ids):
        self.targets[env_ids] = positions


@pytest.mark.parametrize("env_ids", [None, [1, 3]])
def test_pregrasp_reset_changes_only_selected_state_and_keeps_defaults(env_ids):
    robot = _Robot()
    env = SimpleNamespace(scene={"robot": robot}, num_envs=4, device="cpu")
    before = robot.data.default_joint_pos.clone()
    rng = torch.get_rng_state().clone()
    reset_franka_pregrasp(env, env_ids, FRANKA_PREGRASP_JOINT_POSITIONS)
    selected = list(range(4)) if env_ids is None else env_ids
    expected = before[selected].clone()
    expected[:, :7] = torch.tensor(FRANKA_PREGRASP_JOINT_POSITIONS)
    torch.testing.assert_close(robot.positions[selected], expected)
    torch.testing.assert_close(robot.targets[selected], expected)
    assert not robot.velocities[selected].any()
    if env_ids is not None:
        assert not robot.positions[[0, 2]].any()
    torch.testing.assert_close(robot.data.default_joint_pos, before)
    assert torch.equal(torch.get_rng_state(), rng)


def test_easy_variant_rejects_wrong_task_and_unknown_preset():
    with pytest.raises(ValueError, match="requires Isaac-Lift"):
        EnvCfg(name="Isaac-Cartpole-v0", variant="lift_easy")
    with pytest.raises(ValueError, match="env.variant"):
        EnvCfg(variant="easy_typo")


def test_cube_reset_range_changes_only_xy_initial_distribution():
    narrow, broad = _native_config(), _native_config()
    for native, half_range in ((narrow, 0.02), (broad, 0.06)):
        cfg = TrainCfg(
            env=EnvCfg(
                name="Isaac-Lift-Cube-Franka-v0",
                variant="lift_easy",
                lift_cube_xy_half_range=half_range,
                add_success_bonus=False,
            ),
            reward=RewardCfg(source="environment"),
        )
        rewards, actions, terminations = (
            native.rewards,
            native.actions,
            native.terminations,
        )
        build_env_cfg(
            cfg, parse_env_cfg=lambda *args, **kwargs: native, use_fabric=True
        )
        assert (native.rewards, native.actions, native.terminations) == (
            rewards,
            actions,
            terminations,
        )
        assert native.episode_length_s == 5
    assert broad.events.reset_object_position.params["pose_range"] == {
        "x": (-0.06, 0.06),
        "y": (-0.06, 0.06),
        "z": (0.0, 0.0),
    }
    assert broad.scene.object.init_state.pos == narrow.scene.object.init_state.pos
    assert vars(broad.commands.object_pose.ranges) == vars(
        narrow.commands.object_pose.ranges
    )
    assert broad.events.reset_pregrasp.params == narrow.events.reset_pregrasp.params


@pytest.mark.parametrize("half_range", [0, -0.01, 0.11, float("nan"), float("inf")])
def test_invalid_cube_reset_ranges_are_rejected(half_range):
    with pytest.raises(ValueError, match="lift_cube_xy_half_range"):
        EnvCfg(variant="lift_easy", lift_cube_xy_half_range=half_range)


def test_cube_reset_range_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="requires env.variant"):
        EnvCfg(variant="standard", lift_cube_xy_half_range=0.06)


@pytest.mark.parametrize(
    "filename, algorithm", [("pmd_easy.yaml", "pmd"), ("ppo_easy.yaml", "ppo")]
)
def test_easy_experiment_configs_load_with_fixed_rewards(filename, algorithm):
    root = Path(__file__).resolve().parents[1]
    cfg, _ = load_experiment_cfg(
        args=argparse.Namespace(
            task="Isaac-Lift-Cube-Franka-v0",
            config=str(root / "configs/lift_cube_franka" / filename),
        ),
        config_root=root / "configs",
    )
    assert cfg.env.variant == "lift_easy"
    assert cfg.algorithm.type == algorithm
    assert cfg.reward.source == "environment"
    assert not cfg.env.reward_curriculum
    if algorithm == "pmd":
        assert cfg.algorithm.train.gamma == 0.98
        assert cfg.algorithm.entropy_rate == 0.006
        assert cfg.algorithm.value.type == "successor"
        assert cfg.algorithm.train.critic_updates_per_transition == 2
        assert cfg.algorithm.policy_improvements_per_batch == 32
    else:
        assert cfg.algorithm.gamma == 0.98
        assert cfg.algorithm.entropy_rate == 0.006
        assert cfg.env.num_envs == 4096
        assert cfg.runner.steps_per_env_per_cycle == 24


@pytest.mark.parametrize("quantile", [0.25, 0.5, 0.75])
def test_easy_successor_sweep_changes_only_control_quantile(quantile):
    root = Path(__file__).resolve().parents[1]
    args = argparse.Namespace(
        task="Isaac-Lift-Cube-Franka-v0",
        config=str(root / "configs/lift_cube_franka/pmd_easy.yaml"),
    )
    baseline, _ = load_experiment_cfg(args=args, config_root=root / "configs")
    args.set = [f"algo.pmd.value.successor.control_quantile={quantile}"]
    variant, _ = load_experiment_cfg(args=args, config_root=root / "configs")
    expected = asdict(baseline)
    expected["algorithm"]["value"]["control_quantile"] = quantile
    assert asdict(variant) == expected


@pytest.mark.parametrize("angles", [(0.0,) * 6, (float("nan"),) * 7])
def test_pregrasp_reset_rejects_invalid_angles(angles):
    env = SimpleNamespace(scene={"robot": _Robot()}, num_envs=4, device="cpu")
    with pytest.raises(ValueError, match="Pre-grasp"):
        reset_franka_pregrasp(env, None, angles)
