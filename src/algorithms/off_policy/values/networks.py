"""The networks a value backend is built from.

One normalized residual tower (the BRO recipe), a scalar Q ensemble of them,
and a successor-feature ensemble that shares a single trunk across heads.
LayerNorm after every linear layer is per-sample, so nothing depends on dataset
statistics, and it is what keeps a value network from extrapolating without
bound once the policy leaves the states its critic was trained on.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from algorithms.off_policy.values.parameter_ops import frozen_target_copy


class _ResidualBlock(nn.Module):
    """``x + f(x)`` for ``[B,W] -> [B,W]`` with normalized, rectified inner layers."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class BroNet(nn.Module):
    """Map ``[B,I]`` to ``[B,O]`` through a normalized residual tower.

    Shapes:
        input: [B, input_dim]
        hidden: [B, width]
        output: [B, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        width: int,
        residual_blocks: int,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, width) <= 0:
            raise ValueError("BroNet dimensions must be positive.")
        if residual_blocks <= 0:
            raise ValueError("residual_blocks must be > 0.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.stem = nn.Sequential(
            nn.Linear(input_dim, width), nn.LayerNorm(width), nn.ReLU()
        )
        self.blocks = nn.Sequential(
            *(_ResidualBlock(width) for _ in range(residual_blocks))
        )
        self.head = nn.Linear(width, output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input [B,{self.input_dim}], got {tuple(values.shape)}."
            )
        return self.head(self.blocks(self.stem(values)))


def reset_module_parameters(module: nn.Module) -> None:
    """Reinitialize every parameter of ``module`` in place.

    Used by periodic resets, which throw away what the networks have learned
    while keeping the replay they learned it from.
    """
    for child in module.modules():
        if child is not module and hasattr(child, "reset_parameters"):
            child.reset_parameters()


class ScalarActionValue(nn.Module):
    """Map observations ``[B,O]`` and actions ``[B,A]`` to values ``[B]``."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        width: int = 512,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim) <= 0:
            raise ValueError("Action-value dimensions must be positive.")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.network = BroNet(
            observation_dim + action_dim,
            1,
            width=width,
            residual_blocks=residual_blocks,
        )

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(
                f"Expected observations [B,{self.observation_dim}], "
                f"got {tuple(observations.shape)}."
            )
        if tuple(actions.shape) != (observations.shape[0], self.action_dim):
            raise ValueError(
                f"Expected actions [B,{self.action_dim}], got {tuple(actions.shape)}."
            )
        return self.network(torch.cat((observations, actions), dim=-1)).squeeze(-1)

    def target_copy(self) -> "ScalarActionValue":
        return frozen_target_copy(self)

    def reset_parameters(self) -> None:
        reset_module_parameters(self.network)


class ScalarActionValueEnsemble(nn.Module):
    """Independent scalar action-value heads, returning all values ``[M,B]``."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        num_heads: int,
        width: int = 512,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0.")
        self.heads = nn.ModuleList(
            [
                ScalarActionValue(
                    observation_dim,
                    action_dim,
                    width=width,
                    residual_blocks=residual_blocks,
                )
                for _ in range(num_heads)
            ]
        )
        self.num_heads = int(num_heads)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([head(observations, actions) for head in self.heads], dim=0)

    def head(
        self, head_index: int, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if not 0 <= head_index < self.num_heads:
            raise ValueError(
                f"head_index must be in [0,{self.num_heads}), got {head_index}."
            )
        return self.heads[head_index](observations, actions)

    def target_copy(self) -> "ScalarActionValueEnsemble":
        return frozen_target_copy(self)

    def reset_parameters(self) -> None:
        """Reinitialize every head independently."""
        for head in self.heads:
            head.reset_parameters()


def _zero_output_layer(head: nn.Module) -> None:
    """Start a successor head at zero, so an untrained head predicts no return."""
    output_layer = head[-1]
    if not isinstance(output_layer, nn.Linear):
        raise TypeError("Successor head must end in a linear output layer.")
    nn.init.zeros_(output_layer.weight)
    nn.init.zeros_(output_layer.bias)


class SuccessorFeatureEnsemble(nn.Module):
    """Shared residual trunk and independent successor-feature heads.

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
        width: int = 512,
        residual_blocks: int = 2,
        latent_dim: int = 128,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(observation_dim, action_dim, feature_dim, num_heads) <= 0:
            raise ValueError("All dimensions and num_heads must be positive.")
        if min(latent_dim, head_hidden_dim) <= 0:
            raise ValueError("latent_dim and head_hidden_dim must be positive.")
        self.trunk = BroNet(
            observation_dim + action_dim,
            latent_dim,
            width=width,
            residual_blocks=residual_blocks,
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, head_hidden_dim),
                    nn.LayerNorm(head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(head_hidden_dim, feature_dim),
                )
                for _ in range(num_heads)
            ]
        )
        for head in self.heads:
            _zero_output_layer(head)
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
        return frozen_target_copy(self)

    def reset_parameters(self) -> None:
        """Reinitialize the trunk and every head, keeping the output scale.

        The scale is a property of the features, not of what the network has
        learned, so a reset leaves it alone; the heads return to their
        zero-initialized output as at construction.
        """
        reset_module_parameters(self.trunk)
        for head in self.heads:
            reset_module_parameters(head)
            _zero_output_layer(head)


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


def successor_return_span(gamma: float, horizon: int | None = None) -> float:
    """Steps a successor sum effectively runs over.

    A discounted sum decays over ``1/(1-gamma)`` steps, which is where it gets
    its scale from. An undiscounted one does not decay at all, so its scale
    comes from the episode instead - and only a task that actually ends has one.
    Asking for the horizon rather than assuming it is what keeps ``gamma=1``
    from silently meaning "infinite".
    """
    if 0.0 <= gamma < 1.0:
        return 1.0 / (1.0 - float(gamma))
    if gamma != 1.0:
        raise ValueError(f"gamma must be in [0,1], got {gamma}.")
    if horizon is None or horizon <= 0:
        raise ValueError(
            "An undiscounted successor sum spans the episode, so gamma=1 needs "
            f"a positive episode horizon; got {horizon}."
        )
    return float(horizon)


def successor_return_scale_from_second_moment(
    feature_second_moment: torch.Tensor,
    *,
    gamma: float,
    minimum_scale: float,
    horizon: int | None = None,
) -> torch.Tensor:
    """Return a per-feature scale ``[D]`` in return units.

    The per-step feature RMS is multiplied by the number of steps the sum
    spans, so a normalized successor output sits near unit scale. Reward
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
    if minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be > 0.")
    span = successor_return_span(float(gamma), horizon)
    return (feature_second_moment.sqrt() * span).clamp_min(minimum_scale)
