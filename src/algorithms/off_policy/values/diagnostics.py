"""Read-only Monte Carlo diagnostics on complete stochastic policy episodes.

These are errors against sampled returns, not errors against an exactly known
conditional expectation. Deterministic rollouts must not be passed as evidence
about a critic trained to evaluate the stochastic policy.
"""

from __future__ import annotations

import torch

from algorithms.off_policy.actor import TanhPolicyProtocol
from algorithms.off_policy.values.backends import SuccessorActionValue
from reward.terminal_success import TERMINAL_SUCCESS_TERM
from storage.episode_batch import EpisodeBatch
from storage.observations import map_observation


def undiscounted_returns(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Valid suffix sums [B,T,...], including the current transition; padding zero."""
    if values.shape[:2] != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Expected values [B,T,...] and matching boolean mask [B,T].")
    valid = mask.reshape(*mask.shape, *([1] * (values.ndim - 2)))
    return values.masked_fill(~valid, 0).flip(1).cumsum(1).flip(1)


def _return_metrics(
    predictions: torch.Tensor, returns: torch.Tensor, *, quantile: float
) -> dict[str, float]:
    """Predictions [M,V], sampled returns [V] -> raw-unit comparison metrics."""
    estimate = torch.quantile(predictions, quantile, dim=0, interpolation="linear")
    residual = estimate - returns
    metrics = {
        "predicted_mean": float(estimate.mean()),
        "sampled_return_mean": float(returns.mean()),
        "bias": float(residual.mean()),
        "sampled_return_rmse": float(residual.square().mean().sqrt()),
        "head_std_mean": float(predictions.std(dim=0, correction=0).mean()),
    }
    for index, head in enumerate(predictions):
        metrics[f"head_{index}_mean"] = float(head.mean())
        metrics[f"head_{index}_bias"] = float((head - returns).mean())
    return metrics


@torch.no_grad()
def successor_value_diagnostics(
    value: SuccessorActionValue,
    policy: TanhPolicyProtocol,
    episodes: EpisodeBatch,
    *,
    entropy_rate: float,
    feature_names: tuple[str, ...],
) -> dict[str, float]:
    """Compare unchanged critics with stochastic rollouts; gamma=1, finite horizon.

    Each valid state-action gets its realized feature/entropy suffix return.
    Report both all-transition and episode-start comparisons, plus terminal
    feature errors (whose targets require neither bootstrap nor future samples).
    No normalization by critic return scales is applied.
    """
    if value.train_cfg.gamma != 1.0:
        raise ValueError("These finite-horizon diagnostics require gamma=1.")
    rollout, mask = episodes.transitions, episodes.mask
    ended = (rollout.terminated | rollout.truncated) & mask
    if bool(rollout.bootstrap_valid[ended].any()):
        raise ValueError("Finite-horizon episode endings must not bootstrap.")
    if len(feature_names) != rollout.features.shape[-1]:
        raise ValueError("Feature names must match the successor basis.")
    observations = rollout.critic_observations[mask]
    actions = rollout.actions[mask]
    predicted_features = value.online(observations, actions)  # [M,V,D]
    feature_returns = undiscounted_returns(rollout.features, mask)[mask]  # [V,D]
    step_dt = value.reward.step_dt.to(predicted_features)
    weights = value.reward.weights.to(predicted_features)
    reward_predictions = step_dt * torch.einsum(
        "mvd,d->mv", predicted_features, weights
    )
    reward_returns = step_dt * (feature_returns @ weights)
    components = {"reward": (reward_predictions, reward_returns)}
    if TERMINAL_SUCCESS_TERM in feature_names:
        column = feature_names.index(TERMINAL_SUCCESS_TERM)
        components["sparse"] = (
            step_dt * predicted_features[:, :, column],
            step_dt * feature_returns[:, column],
        )
        sparse_prediction, sparse_return = components["sparse"]
        components["learned"] = (
            reward_predictions - weights[column] * sparse_prediction,
            reward_returns - weights[column] * sparse_return,
        )
    if value.entropy is not None:
        policy_obs = map_observation(
            rollout.policy_observations, lambda leaf: leaf[mask]
        )
        entropy_steps = torch.zeros_like(rollout.environment_rewards)
        entropy_steps[mask] = -step_dt * policy.log_probability(policy_obs, actions)
        entropy_suffix = undiscounted_returns(entropy_steps, mask)
        # Q_H excludes the current action's entropy; terminal Q_H is exactly 0.
        future_entropy = torch.zeros_like(entropy_suffix)
        future_entropy[:, :-1] = entropy_suffix[:, 1:]
        entropy_prediction = value.entropy.value(observations, actions)[None, :]
        components["entropy"] = (entropy_prediction, future_entropy[mask])
        components["soft"] = (
            reward_predictions + entropy_rate * entropy_prediction,
            reward_returns + entropy_rate * future_entropy[mask],
        )
    start = torch.zeros_like(mask)
    start[:, 0] = True
    metrics = {}
    for component, (predictions, returns) in components.items():
        for scope, selected in (("all", mask[mask]), ("start", start[mask])):
            comparison = _return_metrics(
                predictions[:, selected],
                returns[selected],
                quantile=value.control_quantile,
            )
            metrics.update(
                {
                    f"{component}/{scope}/{name}": result
                    for name, result in comparison.items()
                }
            )
    terminal = ended[mask]
    residual = predicted_features - feature_returns[None, :, :]
    for column, name in enumerate(feature_names):
        metrics[f"successor/{name}/bias"] = float(residual[:, :, column].mean())
        metrics[f"successor/{name}/sampled_return_rmse"] = float(
            residual[:, :, column].square().mean().sqrt()
        )
        metrics[f"successor/{name}/terminal_rmse"] = float(
            residual[:, terminal, column].square().mean().sqrt()
        )
    return metrics
