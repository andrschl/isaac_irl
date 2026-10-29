from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from algorithms.off_policy.actor import TanhPolicyProtocol
from config.schema import BCCfg
from data import load_obs_action_episodes, split_holdout_episodes, subset_episodes
from interfaces import RuntimeContext
from storage.expert_dataset import ExpertObsActionDataset
from storage.observations import Observation

if TYPE_CHECKING:
    from rsl_rl.models import MLPModel


class BC:
    """Compute a behavioral-cloning loss against an rsl_rl actor.

    BC is a *loss provider*, not a standalone trainer. It holds a reference to
    a shared :class:`storage.expert_dataset.ExpertObsActionDataset` and exposes
    :meth:`compute_loss` so an outer optimizer (typically PPO's, via
    :class:`algorithms.ppo_with_bc.PPOWithBC`) can incorporate the BC gradient
    into its update.
    """

    def __init__(
        self,
        *,
        cfg: BCCfg,
        storage: ExpertObsActionDataset,
        val_storage: ExpertObsActionDataset | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if cfg.loss_type not in {"nll", "mse"}:
            raise ValueError(
                f"`cfg.loss_type` must be 'nll' or 'mse', got {cfg.loss_type!r}."
            )
        if not (0.0 <= cfg.weight <= 1.0):
            raise ValueError(f"`cfg.weight` must be in [0, 1], got {cfg.weight}.")
        if cfg.batch_size <= 0:
            raise ValueError(f"`cfg.batch_size` must be > 0, got {cfg.batch_size}.")

        self.cfg = cfg
        self.storage = storage
        # Held-out (obs, action) pairs for validation logging only; never
        # trained on. None when no train/val split was configured.
        self.val_storage = val_storage
        self.device = torch.device(device)

    @property
    def weight(self) -> float:
        return self.cfg.weight

    def _compute_loss(
        self, actor: MLPModel, storage: ExpertObsActionDataset
    ) -> torch.Tensor:
        """Sample a minibatch from ``storage`` and return the BC loss."""
        observations, expert_actions = storage.sample(self.cfg.batch_size, self.device)
        return self._loss(actor, observations, expert_actions)

    def _loss(
        self,
        actor: MLPModel,
        observations: Observation,
        expert_actions: torch.Tensor,
    ) -> torch.Tensor:
        """BC loss of ``actor`` on given pairs. Shapes: actions [B, A] -> []."""
        if self.cfg.loss_type == "nll":
            actor(observations, stochastic_output=True)
            # GaussianDistribution.log_prob already sums over the action dim.
            return -actor.get_output_log_prob(expert_actions).mean()
        return F.mse_loss(actor(observations), expert_actions)

    def compute_loss(self, actor: MLPModel) -> torch.Tensor:
        """Sample a training minibatch and return the BC loss.

        Args:
            actor: rsl_rl 5.x ``MLPModel`` exposing
                ``forward(obs, stochastic_output=True)`` (to populate the
                output distribution) and ``get_output_log_prob(actions)``.

        Shapes:
            sampled obs minibatch: nested dict; leaves ``[B, *leaf]``
            sampled expert actions: ``[B, A]``

        Returns:
            Scalar loss tensor (shape ``[]``). Gradients flow into
            ``actor.parameters()``; the caller owns
            ``zero_grad`` / ``backward`` / ``step``.
        """
        return self._compute_loss(actor, self.storage)

    def compute_policy_loss(self, policy: TanhPolicyProtocol) -> torch.Tensor:
        """Return BC loss for the bounded SAC/PMD policy.

        Shapes:
            sampled observations: nested dict; leaves ``[B, *leaf]``
            expert actions: ``[B, A]``
        """
        observations, expert_actions = self.storage.sample(
            self.cfg.batch_size, self.device
        )
        _validate_bounded_expert_actions(
            expert_actions, policy, require_interior=self.cfg.loss_type == "nll"
        )
        if self.cfg.loss_type == "nll":
            return -policy.log_probability(observations, expert_actions).mean()
        predicted_actions = policy.deterministic_action(observations)
        return F.mse_loss(predicted_actions, expert_actions)

    def compute_validation_loss(self, actor: MLPModel) -> float | None:
        """Return the BC loss on the held-out validation set, or ``None``.

        ``None`` when no validation split was configured (``val_storage`` is
        empty/absent). Detached and run under ``torch.no_grad()``, so it never
        contributes a gradient. The caller is responsible for putting the actor
        in eval mode if desired.
        """
        if self.val_storage is None or len(self.val_storage) == 0:
            return None
        observations, expert_actions = self.val_storage.all_pairs(self.device)
        with torch.no_grad():
            return float(self._loss(actor, observations, expert_actions).item())


def _validate_bounded_expert_actions(
    expert_actions: torch.Tensor,
    policy: TanhPolicyProtocol,
    *,
    require_interior: bool = False,
) -> None:
    """Reject expert actions a bounded policy could never have produced.

    A tanh policy assigns them no density: the log-probability path has to
    clamp before inverting tanh, which silently turns every out-of-range action
    into the same boundary action.
    """
    action_bound = float(policy.model.action_bound)
    largest = float(expert_actions.abs().max().item())
    if largest > action_bound * (1.0 + 1.0e-6):
        raise ValueError(
            f"Expert actions reach {largest:.4f}, outside the policy action "
            f"bound {action_bound:.4f}. Record demonstrations with the same "
            "action clip as the policy bound, or raise policy.action_bound."
        )
    if require_interior and largest >= action_bound * (1.0 - 1.0e-6):
        raise ValueError(
            "Tanh-policy NLL is undefined for an expert action at the clipped "
            f"bound +/-{action_bound:g}. Use action MSE or fit the spread on "
            "interior actions with spread calibration."
        )


def build_bc_alg(
    *,
    cfg: BCCfg,
    expert_data_path: str,
    ctx: RuntimeContext,
    device: str | torch.device,
    expert_num_trajectories: int | None = None,
    expert_subset_strategy: str = "first",
    seed: int = 42,
) -> BC | None:
    """Return configured BC when ``cfg.weight > 0``, else ``None``.

    Loads (obs, actions) demos from ``expert_data_path``, optionally subsets
    them, splits off a held-out validation set per ``cfg.val_fraction``, and
    wraps the result in ``ExpertObsActionDataset`` instances. Returns ``None`` when
    BC is disabled (``cfg.weight <= 0``) so the caller can short-circuit.
    """
    if cfg.weight <= 0.0:
        return None
    if not expert_data_path:
        raise ValueError(
            "`bc.weight > 0` requires expert_data_path to point at an HDF5 file "
            "with (obs, actions) demos."
        )

    obs_action_episodes = load_obs_action_episodes(expert_data_path)
    if expert_num_trajectories is not None:
        total = len(obs_action_episodes)
        obs_action_episodes = subset_episodes(
            obs_action_episodes,
            int(expert_num_trajectories),
            expert_subset_strategy,
            seed,
        )
        print(
            f"[INFO] BC expert subset: {len(obs_action_episodes)}/{total} "
            f"trajectories (strategy={expert_subset_strategy})"
        )

    train_episodes, val_episodes = split_holdout_episodes(
        obs_action_episodes, cfg.val_fraction, seed
    )
    train_storage = ExpertObsActionDataset(train_episodes)
    val_storage = ExpertObsActionDataset(val_episodes) if val_episodes else None
    val_summary = (
        f", {len(val_episodes)} val episodes ({len(val_storage)} pairs)"
        if val_storage is not None
        else " (no validation split)"
    )
    print(
        f"[INFO] BC dataset: {len(train_episodes)} train episodes "
        f"({len(train_storage)} pairs){val_summary}."
    )
    return BC(cfg=cfg, storage=train_storage, val_storage=val_storage, device=device)
