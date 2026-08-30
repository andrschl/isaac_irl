import pytest

pytest.importorskip("torch")

import algorithms
import collectors
import reward_features
import storage
import utils
from algorithms import IrlAlgorithm as IrlAlgorithmFromPackage
from algorithms import IRLCfg as IRLCfgFromPackage
from algorithms.irl import IrlAlgorithm as IrlAlgorithmFromModule
from algorithms.irl import IRLCfg as IRLCfgFromModule
from reward_features import ManagerBasedFeatureCfg as FeatureCfgFromPackage
from reward_features import ManagerBasedFeatureSource as FeatureSourceFromPackage
from reward_features import manager_based_reward_features as FeatureFnFromPackage
from reward_features.manager_based import ManagerBasedFeatureCfg as FeatureCfgFromModule
from reward_features.manager_based import (
    ManagerBasedFeatureSource as FeatureSourceFromModule,
)
from reward_features.manager_based import (
    manager_based_reward_features as FeatureFnFromModule,
)
from reward_model import RewardModelCfg as RewardCfgFromPackage
from reward_model.dense import RewardModelCfg as RewardCfgFromModule
from runner import IrlRunner as RunnerFromPackage
from runner import IrlRunnerCfg as RunnerCfgFromPackage
from runner.runner import IrlRunner as RunnerFromModule
from runner.runner import IrlRunnerCfg as RunnerCfgFromModule
from storage import FeatureBufCfg as FeatureBufCfgFromPackage
from storage import FeatureTrajectoryBuffer as FeatureBufferFromPackage
from storage.feature_storage import FeatureBufCfg as FeatureBufCfgFromModule
from storage.feature_storage import FeatureTrajectoryBuffer as FeatureBufferFromModule
from utils import RuntimeContext as RuntimeContextFromPackage
from utils.runtime_context import RuntimeContext as RuntimeContextFromModule


def test_algorithm_import_surfaces():
    assert IrlAlgorithmFromPackage is IrlAlgorithmFromModule
    assert IRLCfgFromPackage is IRLCfgFromModule


def test_reward_import_surfaces():
    assert RewardCfgFromPackage is RewardCfgFromModule


def test_runner_import_surfaces():
    assert RunnerFromPackage is RunnerFromModule
    assert RunnerCfgFromPackage is RunnerCfgFromModule


def test_storage_import_surfaces():
    assert FeatureBufCfgFromPackage is FeatureBufCfgFromModule
    assert FeatureBufferFromPackage is FeatureBufferFromModule


def test_reward_features_import_surfaces():
    assert FeatureCfgFromPackage is FeatureCfgFromModule
    assert FeatureSourceFromPackage is FeatureSourceFromModule
    assert FeatureFnFromPackage is FeatureFnFromModule


def test_utils_import_surface():
    assert RuntimeContextFromPackage is RuntimeContextFromModule


def test_collectors_package_is_importable_and_advertises_public_api():
    assert "RobomimicDataCollector" in collectors.__all__
    assert set(algorithms.__all__) == {
        "BC",
        "BCCfg",
        "ExponentialTarget",
        "IRLCfg",
        "IrlAlgorithm",
        "Pmd",
        "Sac",
        "ScalarActionValue",
        "ScalarActionValueEnsemble",
        "SuccessorFeatureEnsemble",
        "adaptive_exponential_target",
        "build_bc_alg",
        "entropy_return_td_target",
        "exponential_log_weights",
        "exponential_target",
        "make_ppo_with_bc_cls",
        "soft_task_q_td_target",
    }
    assert set(storage.__all__) == {
        "FeatureBufCfg",
        "FeatureTrajectoryBuffer",
        "ObsActionBufCfg",
        "ObsActionBuffer",
        "TransitionBatch",
        "TransitionReplayBuffer",
        "TransitionRollout",
    }
    assert "ManagerBasedFeatureCfg" in reward_features.__all__
    assert set(utils.__all__) == {"RuntimeContext", "dump_run_configs"}


def test_actor_initialization_reads_an_off_policy_checkpoint(tmp_path) -> None:
    """A run must be able to start from a policy this repository trained."""
    torch = pytest.importorskip("torch")
    from adapters.rsl_rl_checkpoints import load_actor_only_checkpoint
    from algorithms.state_dependent_tanh_actor import StateDependentTanhGaussianActor

    trained = StateDependentTanhGaussianActor(3, 2, hidden_dims=(8,))
    checkpoint = tmp_path / "model_10.pt"
    torch.save(
        {"rl_alg_state": {"learner": {"policy": trained.state_dict()}}}, checkpoint
    )
    fresh = StateDependentTanhGaussianActor(3, 2, hidden_dims=(8,))

    load_actor_only_checkpoint(fresh, str(checkpoint), device="cpu")

    torch.testing.assert_close(fresh.mean_head.weight, trained.mean_head.weight)


def test_actor_initialization_rejects_a_checkpoint_without_an_actor(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from adapters.rsl_rl_checkpoints import load_actor_only_checkpoint
    from algorithms.state_dependent_tanh_actor import StateDependentTanhGaussianActor

    checkpoint = tmp_path / "model_10.pt"
    torch.save({"rl_alg_state": {"learner": {"value": {}}}}, checkpoint)

    with pytest.raises(ValueError, match="no actor in the checkpoint"):
        load_actor_only_checkpoint(
            StateDependentTanhGaussianActor(3, 2, hidden_dims=(8,)),
            str(checkpoint),
            device="cpu",
        )
