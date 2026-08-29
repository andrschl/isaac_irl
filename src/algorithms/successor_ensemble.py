"""Bootstrapped successor-feature ensemble with a shared representation."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class SuccessorFeatureEnsemble(nn.Module):
    """Shared trunk and independent nonlinear successor-feature heads.

    Shapes:
        observations: [B,O]
        actions: [B,A]
        all outputs: [M,B,D]
        selected outputs: [B,D]
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        feature_dim: int,
        *,
        num_heads: int = 5,
        trunk_hidden_dims: tuple[int, ...] = (256, 256),
        latent_dim: int = 256,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim, feature_dim, num_heads) <= 0:
            raise ValueError("All dimensions and num_heads must be positive.")
        if not trunk_hidden_dims or any(value <= 0 for value in trunk_hidden_dims):
            raise ValueError("trunk_hidden_dims must contain positive dimensions.")
        trunk_dims = (observation_dim + action_dim, *trunk_hidden_dims, latent_dim)
        trunk_layers: list[nn.Module] = []
        for input_dim, output_dim in zip(trunk_dims[:-1], trunk_dims[1:], strict=True):
            trunk_layers.extend((nn.Linear(input_dim, output_dim), nn.ELU()))
        self.trunk = nn.Sequential(*trunk_layers)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, head_hidden_dim),
                    nn.ELU(),
                    nn.Linear(head_hidden_dim, feature_dim),
                )
                for _ in range(num_heads)
            ]
        )
        for head in self.heads:
            output_layer = head[-1]
            if not isinstance(output_layer, nn.Linear):
                raise TypeError("Successor head must end in a linear output layer.")
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.register_buffer("output_scale", torch.ones(feature_dim))

    def latent(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(
                f"Expected observations [B,{self.observation_dim}], "
                f"got {tuple(observations.shape)}."
            )
        if actions.ndim != 2 or actions.shape != (
            observations.shape[0],
            self.action_dim,
        ):
            raise ValueError(
                f"Expected actions [B,{self.action_dim}], got {tuple(actions.shape)}."
            )
        return self.trunk(torch.cat((observations, actions), dim=-1))

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        latent = self.latent(observations, actions)
        normalized = torch.stack([head(latent) for head in self.heads], dim=0)
        return normalized * self.output_scale.reshape(1, 1, self.feature_dim)

    def head(
        self, head_index: int, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return self.apply_head(head_index, self.latent(observations, actions))

    def apply_head(self, head_index: int, latent: torch.Tensor) -> torch.Tensor:
        """Score one head on an already-computed latent ``[B,L] -> [B,D]``.

        The trunk is shared, so a multi-head update evaluates it once for the
        whole batch and hands each head its own slice.
        """
        if not 0 <= head_index < self.num_heads:
            raise ValueError(
                f"head_index must be in [0,{self.num_heads}), got {head_index}."
            )
        normalized = self.heads[head_index](latent)
        return normalized * self.output_scale.reshape(1, self.feature_dim)

    def mean(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(observations, actions).mean(dim=0)

    @torch.no_grad()
    def set_output_scale(
        self, scale: torch.Tensor, *, preserve_predictions: bool = False
    ) -> None:
        """Set positive raw-unit scale [D] for normalized network outputs.

        A prediction is ``head_output * output_scale``, so rescaling every head's
        output layer by ``old / new`` leaves the represented function untouched
        while changing the magnitude the network has to produce. That is what
        makes a later refit safe: only the conditioning moves, not the values.
        """
        scale_tensor = torch.as_tensor(
            scale,
            device=self.output_scale.device,
            dtype=self.output_scale.dtype,
        )
        if tuple(scale_tensor.shape) != (self.feature_dim,):
            raise ValueError(
                f"Expected successor output scale [{self.feature_dim}], "
                f"got {tuple(scale_tensor.shape)}."
            )
        if not torch.isfinite(scale_tensor).all() or torch.any(scale_tensor <= 0.0):
            raise ValueError("Successor output scale must be finite and positive.")
        if preserve_predictions:
            ratio = self.output_scale / scale_tensor
            for head in self.heads:
                output_layer = head[-1]
                # weight is [D, H] and bias [D]: one ratio per output feature.
                output_layer.weight.mul_(ratio.unsqueeze(1))
                output_layer.bias.mul_(ratio)
        self.output_scale.copy_(scale_tensor)

    def target_copy(self) -> "SuccessorFeatureEnsemble":
        target = copy.deepcopy(self)
        target.requires_grad_(False)
        target.eval()
        return target


def successor_ensemble_td_target(
    features: torch.Tensor,
    next_successors: torch.Tensor,
    bootstrap_valid: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Build head-local vector Bellman targets.

    Args:
        features: immediate raw features ``[B,D]``.
        next_successors: output of the same target head ``[B,D]``.
        bootstrap_valid: whether a true next state exists ``[B]``.

    Returns:
        Successor targets ``[B,D]``. Reward parameters never enter this target.
    """
    if features.ndim != 2 or next_successors.shape != features.shape:
        raise ValueError(
            "Expected matching features and next_successors [B,D], got "
            f"{tuple(features.shape)} and {tuple(next_successors.shape)}."
        )
    if tuple(bootstrap_valid.shape) != (features.shape[0],):
        raise ValueError(
            f"Expected bootstrap_valid [B], got {tuple(bootstrap_valid.shape)}."
        )
    if bootstrap_valid.dtype != torch.bool:
        raise TypeError("bootstrap_valid must have dtype bool.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0,1].")
    mask = bootstrap_valid.to(dtype=features.dtype).unsqueeze(-1)
    return features + float(gamma) * mask * next_successors


def successor_return_scale_from_second_moment(
    feature_second_moment: torch.Tensor,
    *,
    gamma: float,
    minimum_scale: float,
) -> torch.Tensor:
    """Return a per-feature scale ``[D]`` in return units.

    The per-step feature RMS is divided by ``1-gamma``, the horizon a discounted
    sum spans, so a normalized successor output sits near unit scale. Reward
    weights never enter, so the scale stays a property of the features alone.
    """
    if feature_second_moment.ndim != 1:
        raise ValueError(
            "Expected a feature second moment [D], got "
            f"{tuple(feature_second_moment.shape)}."
        )
    if not torch.isfinite(feature_second_moment).all():
        raise ValueError("Cannot fit successor scale on non-finite features.")
    if torch.any(feature_second_moment < 0.0):
        raise ValueError("A feature second moment cannot be negative.")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0,1) when estimating return scale.")
    if minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be > 0.")
    return (feature_second_moment.sqrt() / (1.0 - float(gamma))).clamp_min(
        minimum_scale
    )


def estimate_successor_return_scale(
    features: torch.Tensor,
    *,
    gamma: float,
    minimum_scale: float,
) -> torch.Tensor:
    """Estimate the per-feature scale ``[D]`` from one batch of features [B,D]."""
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"Expected non-empty features [B,D], got {features.shape}.")
    return successor_return_scale_from_second_moment(
        features.square().mean(dim=0),
        gamma=gamma,
        minimum_scale=minimum_scale,
    )


@torch.no_grad()
def update_ensemble_target(
    source: SuccessorFeatureEnsemble,
    target: SuccessorFeatureEnsemble,
    coefficient: float,
) -> None:
    """EMA-update the shared target trunk and every paired target head."""
    if not 0.0 < coefficient <= 1.0:
        raise ValueError("coefficient must be in (0,1].")
    for source_parameter, target_parameter in zip(
        source.parameters(), target.parameters(), strict=True
    ):
        target_parameter.lerp_(source_parameter, coefficient)
