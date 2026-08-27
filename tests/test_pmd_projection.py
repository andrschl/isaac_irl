from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from algorithms.off_policy_common import latent_policy_kl  # noqa: E402
from algorithms.pmd import Pmd  # noqa: E402
from algorithms.state_dependent_tanh_actor import (  # noqa: E402
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
)
from algorithms.successor_ensemble import RunningStandardizer  # noqa: E402
from algorithms.value_backends import ValueEstimate  # noqa: E402
from config.off_policy import PmdCfg  # noqa: E402
from conftest import build_replay, build_rollout  # noqa: E402


PREFERRED_ACTION = torch.tensor([0.9, -0.6])


class _QuadraticValue:
    """Stationary Q(s,a) = -||a - a*||^2, so the optimal policy is known."""

    num_heads = 1

    def fit_feature_scale(self, rollout: Any) -> None:
        del rollout

    def update(self, replay: Any, **kwargs: Any) -> dict[str, float]:
        del replay, kwargs
        return {"DirectQ/td_loss": 0.0}

    def mean(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        *,
        entropy_temperature: float,
    ) -> ValueEstimate:
        del observations, entropy_temperature
        q = -(actions - PREFERRED_ACTION.to(actions)).square().sum(dim=-1)
        return ValueEstimate(q=q, reward_q=q, entropy_q=torch.zeros_like(q))

    def modules(self) -> tuple:
        return ()

    def save_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any], *, load_optimizer: bool) -> None:
        del state, load_optimizer


class _ConstantBehaviorCloning:
    """BC regularizer pulling every action towards zero."""

    weight = 0.5

    def compute_policy_loss(self, policy: Any) -> torch.Tensor:
        observations = torch.zeros(8, 3)
        return policy.deterministic_action(observations).square().mean()


def _learner(cfg: PmdCfg | None = None, bc: Any = None) -> Pmd:
    torch.manual_seed(0)
    cfg = cfg or PmdCfg(
        actor_batch_size=64,
        candidate_actions=16,
        projection_steps=6,
        actor_learning_rate=1.0e-2,
    )
    normalizer = RunningStandardizer(3, epsilon=1.0e-6, clip=10.0)
    normalizer.update(torch.randn(128, 3))
    model = StateDependentTanhGaussianActor(
        3,
        2,
        hidden_dims=(16,),
        action_bound=1.0,
        initial_std=1.0,
        minimum_log_std=float(torch.tensor(cfg.policy.minimum_std).log()),
        observation_normalizer=normalizer,
    )
    return Pmd(
        StateDependentTanhPolicy(model),
        _QuadraticValue(),
        actor_observation_normalizer=normalizer,
        critic_observation_normalizer=RunningStandardizer(3, epsilon=1.0e-6, clip=10.0),
        cfg=cfg,
        device=torch.device("cpu"),
        bc=bc,
    )


def _replay():
    return build_replay([build_rollout(num_envs=8, horizon=8, action_dim=2, seed=3)])


def _mean_q(policy) -> float:
    observations = torch.randn(256, 3, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        actions = policy.deterministic_action(observations)
    return float(-(actions - PREFERRED_ACTION).square().sum(dim=-1).mean().item())


def test_projection_respects_the_actor_kl_bound() -> None:
    learner = _learner()
    replay = _replay()
    reference = learner.previous_policy
    observations = torch.randn(128, 3)

    metrics = learner.actor_update(replay)

    assert metrics["PMD/actor_projection_kl"] > 0.0
    assert metrics["PMD/actor_projection_kl"] <= learner.cfg.actor_projection_kl
    # States outside the projection batch may differ, but not by orders.
    held_out_kl = float(
        latent_policy_kl(reference, learner.policy, observations).item()
    )
    assert held_out_kl <= 5.0 * learner.cfg.actor_projection_kl
    assert 1 <= metrics["PMD/projection_steps_taken"] <= learner.cfg.projection_steps


def test_an_oversized_optimizer_step_is_backtracked_into_the_trust_region() -> None:
    learner = _learner(
        PmdCfg(
            actor_batch_size=64,
            candidate_actions=16,
            projection_steps=4,
            actor_learning_rate=1.0,
        )
    )
    replay = _replay()

    metrics = learner.actor_update(replay)

    assert metrics["PMD/projection_backtracks"] > 0.0
    assert metrics["PMD/actor_projection_kl"] <= learner.cfg.actor_projection_kl


def test_repeated_projections_improve_the_action_value() -> None:
    learner = _learner()
    replay = _replay()
    before = _mean_q(learner.policy)

    for _ in range(25):
        learner.actor_update(replay)

    assert _mean_q(learner.policy) > before + 0.1
    assert learner.policy_version == 25


def test_each_projection_refreshes_the_frozen_policy() -> None:
    learner = _learner()
    replay = _replay()
    first_reference = learner.previous_policy

    learner.actor_update(replay)

    assert learner.previous_policy is not first_reference
    observations = torch.randn(32, 3)
    torch.testing.assert_close(
        learner.previous_policy.deterministic_action(observations),
        learner.policy.deterministic_action(observations),
    )


def test_projection_reports_the_behavioral_cloning_contribution() -> None:
    learner = _learner(bc=_ConstantBehaviorCloning())
    replay = _replay()

    metrics = learner.actor_update(replay)

    assert metrics["Actor/bc_weight"] == pytest.approx(0.5)
    assert metrics["Actor/bc_loss"] > 0.0
    assert set(metrics) >= {
        "Actor/loss",
        "Actor/rl_loss",
        "Actor/gradient_norm",
        "Actor/parameter_delta",
        "Actor/policy_std_mean",
        "PMD/q_coefficient",
        "PMD/effective_sample_size",
        "Value/q_mean",
    }


def test_projection_moves_the_actor_every_update() -> None:
    learner = _learner()
    replay = _replay()

    for _ in range(5):
        metrics = learner.actor_update(replay)
        assert metrics["Actor/parameter_delta"] > 0.0
        assert metrics["PMD/projection_steps_taken"] >= 1.0
