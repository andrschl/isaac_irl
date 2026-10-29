"""All four Lift screen arms have one matched PMD/BC data budget."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from config import load_experiment_cfg

ROOT = Path(__file__).resolve().parents[1]
TASK_FEATURES = (
    "reaching_object",
    "lifting_object",
    "object_goal_tracking",
    "object_goal_tracking_fine_grained",
)


def _screen_cfg(source: str, *, learned_success: bool):
    overrides = [
        "env.lift_cube_xy_half_range=0.06",
        "algo.pmd.policy.std_parameterization=smooth_bounded",
        "algo.pmd.policy.minimum_std=0.005",
        "algo.pmd.train.initial_critic_updates=10000",
        "algo.pmd.actor_learning_rate=3e-6",
        "algo.pmd.policy_improvements_per_batch=1",
        "algo.pmd.train.replay.demonstration_path=demos.hdf5",
        "runner.transition_budget=40000",
        "runner.total_transition_budget=48000",
        "runner.save_interval=1",
    ]
    if source != "environment":
        overrides.extend(
            [
                "irl.solver=exact_linear",
                "irl.gradient_steps=0",
                "irl.weight_decay=0",
                *[f"reward.learned_feature_scales.{name}=5" for name in TASK_FEATURES],
            ]
        )
        if learned_success:
            overrides.append("reward.learned_feature_scales.terminal_success=1")
    args = SimpleNamespace(
        task="Isaac-Lift-Cube-Franka-v0",
        config=str(ROOT / "configs/lift_cube_franka/irl_bc.yaml"),
        expert_data_path="demos.hdf5",
        reward_source=source,
        bc_weight=0.99,
        max_iterations=5,
        init_policy_checkpoint="bc.pt",
        set=overrides,
    )
    return load_experiment_cfg(args=args, config_root=ROOT / "configs")[0]


@pytest.mark.parametrize(
    "source,learned_success,expected_names",
    [
        ("learned", False, TASK_FEATURES),
        ("learned", True, (*TASK_FEATURES, "terminal_success")),
        ("environment", False, ()),
        ("environment_plus_learned", False, TASK_FEATURES),
    ],
)
def test_matched_screen_configuration(source, learned_success, expected_names):
    cfg = _screen_cfg(source, learned_success=learned_success)
    assert cfg.bc.weight == 0.99
    assert cfg.algorithm.train.gamma == 1.0
    assert cfg.algorithm.train.initial_critic_updates == 10000
    assert cfg.algorithm.train.critic_updates_per_transition == 2.0
    assert cfg.algorithm.policy_improvements_per_batch == 1
    assert cfg.algorithm.actor_learning_rate == 3e-6
    assert cfg.runner.transition_budget == 40000
    assert cfg.runner.total_transition_budget == 48000
    assert cfg.env.is_finite_horizon and cfg.env.terminal_success_reward
    scales = cfg.reward.learned_feature_scales or {}
    assert tuple(scales) == expected_names
    if source == "environment":
        assert not cfg.reward.learns_a_reward
    else:
        assert cfg.irl.solver == "exact_linear"
