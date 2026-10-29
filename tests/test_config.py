"""Loading an experiment: strict YAML, discriminated algorithms, CLI overrides."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from config import build_dataclass, load_experiment_cfg, parse_overrides, task_slug
from config.schema import (
    DirectQCfg,
    EnvCfg,
    IRLCfg,
    OffPolicyTrainCfg,
    PmdCfg,
    PpoAlgoCfg,
    RewardCfg,
    RewardModelCfg,
    RunnerCfg,
    SuccessorQCfg,
    TrainCfg,
)


def _write_config(root: Path, payload: dict, *, task: str = "lift_cube_franka") -> Path:
    directory = root / task
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        task="Isaac-Lift-Cube-Franka-v0",
        config=None,
        set=[],
        expert_data_path="demos.hdf5",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_task_slug_picks_the_config_directory() -> None:
    assert task_slug("Isaac-Lift-Cube-Franka-v0") == "lift_cube_franka"
    assert task_slug("Isaac-Cartpole-v0") == "cartpole"


def test_native_ppo_benchmark_preserves_curriculum_and_short_rollouts():
    root = Path(__file__).resolve().parents[1]
    cfg, _ = load_experiment_cfg(
        args=_args(config=str(root / "configs/lift_cube_franka/ppo.yaml")),
        config_root=root / "configs",
    )
    assert cfg.is_on_policy and cfg.env.reward_curriculum
    assert cfg.env.num_envs == 4096
    assert cfg.runner.steps_per_env_per_cycle == 24
    assert cfg.runner.validation_interval == 0


@pytest.mark.parametrize(
    "algorithm,reward",
    [
        (PmdCfg(), RewardCfg(source="environment")),
        (PpoAlgoCfg(), RewardCfg(source="learned")),
        (PpoAlgoCfg(), RewardCfg(source="environment", fit_diagnostic=True)),
    ],
)
def test_curriculum_rejects_nonstationary_feature_reward_training(algorithm, reward):
    with pytest.raises(ValueError, match="reward_curriculum=true requires PPO"):
        TrainCfg(env=EnvCfg(reward_curriculum=True), algorithm=algorithm, reward=reward)


def test_curriculum_is_an_explicit_environment_reward_ppo_option():
    cfg = TrainCfg(
        env=EnvCfg(reward_curriculum=True), reward=RewardCfg(source="environment")
    )
    assert cfg.env.reward_curriculum


def test_exact_linear_solver_requires_zero_gradient_steps_and_weight_decay():
    with pytest.raises(ValueError, match="gradient_steps=0 and weight_decay=0"):
        IRLCfg(solver="exact_linear")
    cfg = IRLCfg(solver="exact_linear", gradient_steps=0, weight_decay=0)
    assert cfg.solver == "exact_linear"


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("inf"), float("nan")])
def test_named_feature_scales_must_be_positive_finite(bad_scale):
    with pytest.raises(ValueError, match="Invalid learned feature scale"):
        RewardCfg(
            source="learned",
            model=RewardModelCfg(is_linear=True),
            learned_feature_scales={"reach": bad_scale},
        )


def test_mixed_reward_cannot_learn_the_fixed_terminal_success():
    with pytest.raises(ValueError, match="do not also learn it"):
        TrainCfg(
            reward=RewardCfg(
                source="environment_plus_learned",
                model=RewardModelCfg(is_linear=True),
                learned_feature_scales={"terminal_success": 1.0},
            )
        )


@pytest.mark.parametrize("old_key", ["rollout_steps", "updates_per_transition"])
def test_old_pmd_collection_settings_are_not_silently_reinterpreted(old_key):
    with pytest.raises(ValueError, match="Unknown key"):
        build_dataclass(OffPolicyTrainCfg, {old_key: 8})


def test_old_policy_cadence_is_rejected():
    with pytest.raises(ValueError, match="Unknown key"):
        build_dataclass(PmdCfg, {"critic_updates_per_projection": 8})


def test_episode_pmd_cannot_pay_for_a_second_irl_rollout():
    with pytest.raises(ValueError, match="imitator_source=collected"):
        TrainCfg(algorithm=PmdCfg(), runner=RunnerCfg(imitator_source="rollout"))


@pytest.mark.parametrize(
    ("task", "expected_bound"),
    [("Isaac-Lift-Cube-Franka-v0", 4.0), ("Isaac-Cartpole-v0", 1.0)],
)
def test_checked_in_pmd_action_bounds_are_task_specific(
    task: str, expected_bound: float
) -> None:
    cfg, _ = load_experiment_cfg(
        args=_args(task=task, reward_source="environment", expert_data_path=None),
        config_root=Path(__file__).resolve().parents[1] / "configs",
    )

    assert isinstance(cfg.algorithm, PmdCfg)
    assert cfg.algorithm.policy.action_bound == expected_bound


def test_unknown_keys_are_rejected_with_the_valid_ones_listed() -> None:
    with pytest.raises(ValueError, match=r"Unknown key\(s\) \['reward_lr'\]"):
        build_dataclass(IRLCfg, {"reward_lr": 1e-3})


def test_nested_sections_build_without_a_parser_per_field(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "seed": 7,
            "runner": {
                "steps_per_env_per_cycle": 8,
                "expert_buffer": {"min_ep_len": 3},
            },
            "bc": {"weight": 0.25, "loss_type": "mse"},
        },
    )

    cfg, task_name = load_experiment_cfg(args=_args(), config_root=tmp_path)

    assert task_name == "Isaac-Lift-Cube-Franka-v0"
    assert cfg.seed == 7
    assert cfg.runner.steps_per_env_per_cycle == 8
    assert cfg.runner.expert_buffer.min_ep_len == 3
    assert (cfg.bc.weight, cfg.bc.loss_type) == (0.25, "mse")


def test_a_stale_key_fails_the_run_instead_of_being_ignored(tmp_path: Path) -> None:
    """The whole point of strict loading: a renamed key is never a silent default."""
    _write_config(tmp_path, {"runner": {"policy_updates_per_cycle": 4}})

    with pytest.raises(ValueError, match="policy_updates_per_cycle"):
        load_experiment_cfg(args=_args(), config_root=tmp_path)


def test_algorithm_type_selects_ppo_or_pmd(tmp_path: Path) -> None:
    _write_config(tmp_path, {"algo": {"type": "pmd", "beta": 0.5}})

    cfg, _ = load_experiment_cfg(args=_args(), config_root=tmp_path)

    assert isinstance(cfg.algorithm, PmdCfg)
    assert cfg.algorithm.beta == 0.5
    assert not cfg.is_on_policy


def test_algorithm_override_selects_that_algorithm_s_own_tuning(tmp_path: Path) -> None:
    """Switching arms must not silently untune the one you switched to.

    A benchmark where one algorithm carries the task's tuning and the other
    falls back to defaults measures the tuning, not the algorithms.
    """
    _write_config(
        tmp_path,
        {
            "algo": {
                "type": "pmd",
                "ppo": {"learning_rate": 1.0e-3, "gamma": 0.99},
                "pmd": {
                    "actor_batch_size": 256,
                    "train": {"episodes_per_batch": 8},
                    "value": {"type": "direct"},
                },
            }
        },
    )

    ppo, _ = load_experiment_cfg(args=_args(algorithm="ppo"), config_root=tmp_path)
    pmd, _ = load_experiment_cfg(args=_args(algorithm="pmd"), config_root=tmp_path)

    assert isinstance(ppo.algorithm, PpoAlgoCfg)
    assert (ppo.algorithm.learning_rate, ppo.algorithm.gamma) == (1.0e-3, 0.99)
    assert isinstance(pmd.algorithm, PmdCfg)
    assert pmd.algorithm.actor_batch_size == 256
    assert pmd.algorithm.train.episodes_per_batch == 8


def test_one_algorithm_s_settings_never_leak_into_another(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {"algo": {"type": "pmd", "pmd": {"actor_batch_size": 256}}},
    )

    cfg, _ = load_experiment_cfg(args=_args(algorithm="ppo"), config_root=tmp_path)

    assert isinstance(cfg.algorithm, PpoAlgoCfg)


def test_value_backend_override_keeps_the_tuned_architecture(tmp_path: Path) -> None:
    """Selecting a backend must not silently discard a task's tuned sizes."""
    _write_config(
        tmp_path,
        {"algo": {"type": "pmd", "value": {"type": "successor", "num_heads": 9}}},
    )

    cfg, _ = load_experiment_cfg(
        args=_args(value_backend="direct"), config_root=tmp_path
    )

    assert isinstance(cfg.algorithm, PmdCfg)
    assert isinstance(cfg.algorithm.value, DirectQCfg)
    assert cfg.algorithm.value.num_heads == 9


def test_successor_backend_is_the_pmd_default(tmp_path: Path) -> None:
    _write_config(tmp_path, {"algo": {"type": "pmd"}})

    cfg, _ = load_experiment_cfg(args=_args(), config_root=tmp_path)

    assert isinstance(cfg.algorithm.value, SuccessorQCfg)


def test_named_backend_overrides_nested_lift_tuning():
    cfg, _ = load_experiment_cfg(
        args=_args(value_backend="direct", reward_source="environment"),
        config_root=Path(__file__).resolve().parents[1] / "configs",
    )
    assert isinstance(cfg.algorithm.value, DirectQCfg)
    assert cfg.algorithm.value.num_heads == 3
    assert cfg.algorithm.value.target_subset_size == 2


@pytest.mark.parametrize("algorithm", ["pmd", "ppo"])
def test_named_learning_rate_overrides_selected_algorithm_tuning(algorithm):
    cfg, _ = load_experiment_cfg(
        args=_args(algorithm=algorithm, learning_rate=0.0007),
        config_root=Path(__file__).resolve().parents[1] / "configs",
    )
    key = "learning_rate" if algorithm == "ppo" else "actor_learning_rate"
    assert getattr(cfg.algorithm, key) == 0.0007


def test_value_backend_flag_cannot_be_silently_ignored_for_ppo(tmp_path):
    _write_config(tmp_path, {"algo": {"type": "ppo"}})
    with pytest.raises(ValueError, match="requires --algorithm pmd"):
        load_experiment_cfg(args=_args(value_backend="direct"), config_root=tmp_path)


def test_successor_control_quantile_loads_from_yaml(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "algo": {
                "type": "pmd",
                "value": {"type": "successor", "control_quantile": 0.2},
            }
        },
    )

    cfg, _ = load_experiment_cfg(args=_args(), config_root=tmp_path)

    assert isinstance(cfg.algorithm.value, SuccessorQCfg)
    assert cfg.algorithm.value.control_quantile == 0.2


def test_stale_successor_control_reduction_is_rejected(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        {
            "algo": {
                "type": "pmd",
                "value": {"type": "successor", "control_reduction": "min"},
            }
        },
    )

    with pytest.raises(ValueError, match="control_reduction"):
        load_experiment_cfg(args=_args(), config_root=tmp_path)


def test_named_flags_override_the_file(tmp_path: Path) -> None:
    _write_config(tmp_path, {"seed": 1, "max_iterations": 10, "bc": {"weight": 0.0}})

    cfg, _ = load_experiment_cfg(
        args=_args(seed=42, max_iterations=99, bc_weight=0.5), config_root=tmp_path
    )

    assert (cfg.seed, cfg.max_iterations, cfg.bc.weight) == (42, 99, 0.5)


def test_set_reaches_any_nested_value(tmp_path: Path) -> None:
    _write_config(tmp_path, {"algo": {"type": "pmd"}})

    cfg, _ = load_experiment_cfg(
        args=_args(
            set=[
                "algo.train.critic_updates_per_batch=250",
                "runner.save_interval=10",
                "runner.transition_budget=null",
            ]
        ),
        config_root=tmp_path,
    )

    assert cfg.algorithm.train.critic_updates_per_batch == 250
    assert cfg.runner.save_interval == 10
    assert cfg.runner.transition_budget is None


def test_transition_utd_config_loads_and_keeps_policy_budget_independent(tmp_path):
    _write_config(tmp_path, {"algo": {"type": "pmd"}})
    cfg, _ = load_experiment_cfg(
        args=_args(
            set=[
                "algo.train.critic_updates_per_batch=null",
                "algo.train.critic_updates_per_transition=2.0",
                "algo.policy_improvements_per_batch=32",
            ]
        ),
        config_root=tmp_path,
    )
    assert cfg.algorithm.train.critic_updates_per_transition == 2.0
    assert cfg.algorithm.train.critic_updates_per_batch is None
    assert cfg.algorithm.policy_improvements_per_batch == 32


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf")])
def test_transition_utd_requires_a_positive_finite_ratio(ratio):
    with pytest.raises(ValueError, match="critic_updates_per_transition"):
        OffPolicyTrainCfg(
            critic_updates_per_batch=None, critic_updates_per_transition=ratio
        )


@pytest.mark.parametrize("fixed,ratio", [(None, None), (1000, 2.0)])
def test_critic_budget_modes_are_mutually_exclusive(fixed, ratio):
    with pytest.raises(ValueError, match="exactly one"):
        OffPolicyTrainCfg(
            critic_updates_per_batch=fixed, critic_updates_per_transition=ratio
        )


def test_utd_guarantees_critic_fitting_before_each_policy_improvement():
    with pytest.raises(ValueError, match="minimum critic budget"):
        PmdCfg(
            train=OffPolicyTrainCfg(
                episodes_per_batch=4,
                critic_updates_per_batch=None,
                critic_updates_per_transition=2.0,
            ),
            policy_improvements_per_batch=9,
        )


@pytest.mark.parametrize(
    "assignment, expected",
    [
        ("a=1", 1),
        ("a=1.5", 1.5),
        ("a=true", True),
        ("a=null", None),
        ("a=cpu", "cpu"),
        ("a=[1, 2]", [1, 2]),
    ],
)
def test_set_parses_scalars_the_way_yaml_would(assignment: str, expected) -> None:
    assert parse_overrides([assignment]) == {"a": expected}


def test_set_requires_a_value() -> None:
    with pytest.raises(ValueError, match="PATH=VALUE"):
        parse_overrides(["algo.beta"])


def test_learning_a_reward_requires_demonstrations(tmp_path: Path) -> None:
    _write_config(tmp_path, {"reward": {"source": "learned"}})

    with pytest.raises(ValueError, match="demonstrations"):
        load_experiment_cfg(args=_args(expert_data_path=None), config_root=tmp_path)


def test_environment_reward_runs_without_demonstrations(tmp_path: Path) -> None:
    _write_config(tmp_path, {"reward": {"source": "environment"}})

    cfg, _ = load_experiment_cfg(
        args=_args(expert_data_path=None), config_root=tmp_path
    )

    assert not cfg.reward.learns_a_reward


def test_a_diagnostic_fit_only_applies_to_the_environment_reward() -> None:
    with pytest.raises(ValueError, match="fit_diagnostic"):
        RewardCfg(source="learned", fit_diagnostic=True)

    assert RewardCfg(source="environment", fit_diagnostic=True).learns_a_reward


def test_gamma_comes_from_whichever_algorithm_is_configured() -> None:
    assert TrainCfg(algorithm=PpoAlgoCfg(gamma=0.9)).gamma == 0.9
    assert TrainCfg(algorithm=PmdCfg()).gamma == PmdCfg().train.gamma


def test_a_missing_config_file_names_the_path_it_looked_for(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="experiment.yaml"):
        load_experiment_cfg(args=_args(), config_root=tmp_path)


# --------------------------------------------- the sparse finite-horizon task


def _terminal_env(**overrides) -> EnvCfg:
    settings = dict(
        name="Isaac-Lift-Cube-Franka-v0",
        terminal_success_reward=True,
        is_finite_horizon=True,
        add_success_bonus=False,
    )
    settings.update(overrides)
    return EnvCfg(**settings)


def test_a_sparse_terminal_reward_needs_the_horizon_it_is_defined_on() -> None:
    """Without it the time limit still bootstraps, so the reward is not sparse."""
    with pytest.raises(ValueError, match="is_finite_horizon must be true"):
        _terminal_env(is_finite_horizon=False)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"add_success_bonus": True}, "add_success_bonus must be false"),
        ({"reward_curriculum": True}, "reward_curriculum must be false"),
    ],
)
def test_a_second_success_signal_is_refused_alongside_the_sparse_one(
    overrides, expected
) -> None:
    with pytest.raises(ValueError, match=expected):
        _terminal_env(**overrides)


def test_the_sparse_task_names_the_environment_it_is_implemented_for() -> None:
    with pytest.raises(ValueError, match="Isaac-Lift-Cube-Franka-v0"):
        _terminal_env(name="Isaac-Cartpole-v0")


def test_holding_a_fixed_coefficient_needs_a_linear_reward() -> None:
    """A fixed and a learned coefficient only add on one named basis."""
    with pytest.raises(ValueError, match="reward.model.is_linear"):
        TrainCfg(
            env=_terminal_env(),
            reward=RewardCfg(source="learned", model=RewardModelCfg(is_linear=False)),
            algorithm=PmdCfg(),
        )


def test_adding_the_environment_reward_to_a_learned_one_needs_a_linear_model() -> None:
    with pytest.raises(ValueError, match="environment_plus_learned"):
        TrainCfg(
            reward=RewardCfg(
                source="environment_plus_learned",
                model=RewardModelCfg(is_linear=False),
            ),
            algorithm=PmdCfg(),
        )


@pytest.mark.parametrize(
    "source", ["environment", "learned", "environment_plus_learned"]
)
def test_every_study_arm_is_a_loadable_configuration(source) -> None:
    cfg = TrainCfg(
        env=_terminal_env(),
        reward=RewardCfg(source=source, model=RewardModelCfg(is_linear=True)),
        algorithm=PmdCfg(),
    )

    assert cfg.env.is_finite_horizon
    assert cfg.reward.learns_a_reward is (source != "environment")


def test_continuing_tasks_keep_bootstrapping_through_their_time_limit() -> None:
    """The default has to stay what every existing configuration assumes."""
    assert EnvCfg().is_finite_horizon is False
    assert EnvCfg().terminal_success_reward is False


@pytest.mark.parametrize(
    "arm, source",
    [
        ("sparse_bc", "environment"),
        ("sparse_irl_bc", "environment_plus_learned"),
        ("irl_bc", "learned"),
    ],
)
def test_the_checked_in_study_arms_differ_only_in_their_reward(arm, source) -> None:
    """Three arms, one protocol: a difference in results is a difference in reward."""
    configs = Path(__file__).resolve().parents[1] / "configs"
    cfg, _ = load_experiment_cfg(
        args=_args(config=str(configs / "lift_cube_franka" / f"{arm}.yaml")),
        config_root=configs,
    )

    assert cfg.reward.source == source
    assert cfg.env.variant == "lift_easy"
    assert cfg.env.terminal_success_reward and cfg.env.is_finite_horizon
    assert cfg.env.success_hold_steps == 25
    assert cfg.algorithm.train.gamma == 1.0
    assert cfg.irl.discount_gamma is None
    assert not cfg.irl.normalize_returns_by_episode_length
    assert cfg.algorithm.train.critic_updates_per_transition == 2.0
    assert cfg.algorithm.policy_improvements_per_batch == 16
    assert cfg.algorithm.projection_steps == 8
    assert cfg.algorithm.value.control_quantile == 0.5
    assert cfg.algorithm.entropy_rate == 0.006
    assert cfg.bc.weight == 0.1 and cfg.bc.loss_type == "mse"
    assert cfg.runner.rl_updates_per_cycle == 1
    assert cfg.runner.reward_updates_per_cycle == 1
    assert cfg.irl.gradient_steps == 8
    # 100 cycles of 32 x 250 is exactly the study's interaction budget.
    assert cfg.runner.transition_budget == 800_000
    assert cfg.max_iterations * cfg.algorithm.train.episodes_per_batch * 250 == 800_000


def test_the_arms_agree_on_everything_except_the_reward() -> None:
    configs = Path(__file__).resolve().parents[1] / "configs" / "lift_cube_franka"
    loaded = {}
    for arm in ("sparse_bc", "sparse_irl_bc", "irl_bc"):
        cfg, _ = load_experiment_cfg(
            args=_args(config=str(configs / f"{arm}.yaml")),
            config_root=configs.parent,
        )
        loaded[arm] = replace(
            cfg,
            experiment_name="",
            reward=replace(cfg.reward, source="learned"),
        )

    assert loaded["sparse_bc"] == loaded["sparse_irl_bc"] == loaded["irl_bc"]


def test_a_run_reads_one_demonstration_set(tmp_path: Path) -> None:
    """Imitating one expert and scoring against another would be invisible."""
    _write_config(
        tmp_path,
        {
            "expert_data": {"path": "a.hdf5"},
            "reward": {"source": "environment"},
            "algo": {
                "type": "pmd",
                "pmd": {"train": {"replay": {"demonstration_path": "b.hdf5"}}},
            },
        },
    )

    with pytest.raises(ValueError, match="name different files"):
        load_experiment_cfg(args=_args(expert_data_path="a.hdf5"), config_root=tmp_path)
