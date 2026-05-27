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
from reward_features import ManagerBasedRewardFeatureEncoder as FeatureEncoderFromPackage
from reward_features import manager_based_reward_features as FeatureFnFromPackage
from reward_features.manager_based import ManagerBasedFeatureCfg as FeatureCfgFromModule
from reward_features.manager_based import ManagerBasedRewardFeatureEncoder as FeatureEncoderFromModule
from reward_features.manager_based import manager_based_reward_features as FeatureFnFromModule
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
    assert FeatureEncoderFromPackage is FeatureEncoderFromModule
    assert FeatureFnFromPackage is FeatureFnFromModule


def test_utils_import_surface():
    assert RuntimeContextFromPackage is RuntimeContextFromModule


def test_collectors_package_is_importable_and_advertises_public_api():
    assert "RobomimicDataCollector" in collectors.__all__
    assert set(algorithms.__all__) == {"BC", "BCCfg", "IRLCfg", "IrlAlgorithm", "make_ppo_with_bc_cls"}
    assert set(storage.__all__) == {
        "FeatureBufCfg",
        "FeatureTrajectoryBuffer",
        "ObsActionBufCfg",
        "ObsActionBuffer",
    }
    assert "ManagerBasedFeatureCfg" in reward_features.__all__
    assert utils.__all__ == ["RuntimeContext"]
