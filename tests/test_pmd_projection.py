from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from conftest import build_replay, build_rollout  # noqa: E402

from algorithms.off_policy.actor import (  # noqa: E402
    StateDependentTanhGaussianActor,
    StateDependentTanhPolicy,
)
from algorithms.off_policy.common import decomposed_policy_kl  # noqa: E402
from algorithms.off_policy.exponential_target import (  # noqa: E402
    adaptive_exponential_target,
)
from algorithms.off_policy.pmd import Pmd  # noqa: E402
from algorithms.off_policy.values.backends import ValueEstimate  # noqa: E402
from config.schema import PmdCfg, TanhGaussianPolicyCfg  # noqa: E402

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
        entropy_rate: float,
    ) -> ValueEstimate:
        del observations, entropy_rate
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


def _learner(
    cfg: PmdCfg | None = None,
    bc: Any = None,
    value: Any = None,
    *,
    step_dt: float = 1.0,
) -> Pmd:
    torch.manual_seed(0)
    cfg = cfg or PmdCfg(
        actor_batch_size=64,
        candidate_actions=16,
        projection_steps=6,
        actor_learning_rate=1.0e-2,
    )
    model = StateDependentTanhGaussianActor(
        3,
        2,
        hidden_dims=(16,),
        action_bound=1.0,
        initial_std=cfg.policy.initial_std,
        minimum_log_std=float(torch.tensor(cfg.policy.minimum_std).log()),
        maximum_log_std=float(torch.tensor(cfg.policy.maximum_std).log()),
    )
    return Pmd(
        StateDependentTanhPolicy(model),
        value or _QuadraticValue(),
        cfg=cfg,
        step_dt=step_dt,
        device=torch.device("cpu"),
        bc=bc,
    )


def test_collection_policy_kl_measures_cumulative_not_last_projection_drift():
    learner = _learner()
    rollout = build_rollout(num_envs=2, horizon=8, rollout_id=0, policy_version=0)
    learner.observe_rollout(rollout)
    replay = build_replay([rollout])
    reference = learner.policy.snapshot()
    for _ in range(3):
        metrics = learner.actor_update(replay)
    mean, spread = decomposed_policy_kl(
        reference, learner.policy, learner._collection_observations
    )
    assert metrics["PMD/collection_policy_kl"] == pytest.approx(float(mean + spread))
    assert metrics["PMD/collection_policy_kl"] > 0
    learner.observe_rollout(rollout)
    assert learner._collection_policy_metrics()["PMD/collection_policy_kl"] == 0


def _replay():
    return build_replay([build_rollout(num_envs=8, horizon=8, action_dim=2, seed=3)])


def _mean_q(policy) -> float:
    observations = torch.randn(256, 3, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        actions = policy.deterministic_action(observations)
    return float(-(actions - PREFERRED_ACTION).square().sum(dim=-1).mean().item())


def test_policy_improvement_includes_current_dt_scaled_entropy() -> None:
    cfg = PmdCfg(
        entropy_rate=0.4,
        actor_batch_size=64,
        candidate_actions=16,
        projection_steps=2,
    )
    learner = _learner(cfg, step_dt=0.05)
    replay = _replay()
    batch = replay.sample(
        cfg.actor_batch_size,
        generator=torch.Generator().manual_seed(11),
        device=torch.device("cpu"),
    )
    candidates = learner._candidate_batch(
        batch.policy_observations,
        batch.critic_observations,
        generator=torch.Generator().manual_seed(13),
    )
    soft_improvement_q = (
        candidates.value.q - 0.05 * 0.4 * candidates.reference_log_probability
    )
    expected = adaptive_exponential_target(
        soft_improvement_q,
        candidates.reference_log_probability,
        beta=cfg.beta,
        target_kl=cfg.candidate_target_kl,
        coefficient_min=cfg.q_coefficient_min,
        coefficient_max=cfg.q_coefficient_max,
        bisection_steps=cfg.coefficient_bisection_steps,
    )

    assert learner.entropy_step_coefficient == pytest.approx(0.02)
    torch.testing.assert_close(candidates.target.probabilities, expected.probabilities)


def test_projection_respects_the_actor_kl_bound() -> None:
    learner = _learner()
    replay = _replay()
    reference = learner.previous_policy
    observations = torch.randn(128, 3)

    metrics = learner.actor_update(replay)

    assert metrics["PMD/actor_projection_kl"] > 0.0
    assert metrics["PMD/actor_projection_kl"] <= learner.cfg.actor_projection_kl
    assert metrics["PMD/actor_spread_kl"] <= learner.cfg.actor_spread_projection_kl
    # States outside the projection batch may differ, but not by orders.
    held_out_kl = float(
        sum(decomposed_policy_kl(reference, learner.policy, observations)).item()
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


def test_projection_bounds_the_spread_separately_from_the_mean() -> None:
    learner = _learner()
    replay = _replay()

    for _ in range(30):
        metrics = learner.actor_update(replay)
        assert metrics["PMD/actor_spread_kl"] <= learner.cfg.actor_spread_projection_kl
        assert metrics["PMD/actor_mean_kl"] + metrics["PMD/actor_spread_kl"] <= (
            learner.cfg.actor_projection_kl
        )


def test_repeated_projections_cannot_inflate_the_policy_spread() -> None:
    learner = _learner()
    replay = _replay()
    observations = torch.randn(128, 3)
    _, initial_std = learner.policy.distribution_parameters(observations)

    for _ in range(30):
        learner.actor_update(replay)

    _, final_std = learner.policy.distribution_parameters(observations)
    growth = float((final_std / initial_std).max().item())
    # A spread bound of 0.002 nats per projection over 30 projections cannot
    # take the policy anywhere near the saturated regime.
    assert growth < 1.5


def test_noise_weighted_projections_stay_inside_the_spread_ceiling() -> None:
    class _NoisyValue(_QuadraticValue):
        """A critic with no signal: weights are pure noise, as early training."""

        def mean(self, observations, actions, *, entropy_rate):
            del observations, entropy_rate
            q = torch.randn(actions.shape[0], generator=None)
            return ValueEstimate(q=q, reward_q=q, entropy_q=torch.zeros_like(q))

    cfg = PmdCfg(
        actor_batch_size=64,
        candidate_actions=16,
        projection_steps=6,
        actor_learning_rate=1.0e-2,
        policy=TanhGaussianPolicyCfg(initial_std=1.0, maximum_std=1.5),
    )
    learner = _learner(cfg, value=_NoisyValue())
    replay = _replay()
    observations = torch.randn(64, 3)

    for _ in range(60):
        learner.actor_update(replay)

    _, std = learner.policy.distribution_parameters(observations)
    assert float(std.max().item()) <= cfg.policy.maximum_std + 1.0e-6
