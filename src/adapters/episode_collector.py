"""Episode-wave collection using one Isaac environment and a frozen actor."""

from __future__ import annotations

from dataclasses import fields
from time import perf_counter
from typing import Callable

import torch

from adapters.transition_collector import TransitionCollector
from storage.episode_batch import EpisodeBatch
from storage.observations import Observation, map_observation
from storage.transition_replay import TransitionRollout


class EpisodeCollector:
    """Collect exactly B first episodes in waves of at most N environments.

    Isaac steps every lane. After an early ending, further transitions in that
    lane are excluded, but their interaction cost is counted. No early-failure
    or timeout flag is changed. With one robot there is no lane overhead.
    """

    def __init__(
        self, step_collector: TransitionCollector, *, episodes_per_batch: int
    ) -> None:
        if episodes_per_batch <= 0:
            raise ValueError("episodes_per_batch must be > 0.")
        self.steps = step_collector
        self.episodes_per_batch = episodes_per_batch
        self.horizon = int(step_collector.env.max_episode_length)
        if self.horizon <= 0:
            raise ValueError("Episode collection requires env.max_episode_length > 0.")
        self._batch_id = 0

    def maximum_transitions(self, episodes: int | None = None) -> int:
        """Worst-case vector-lane interactions for a complete batch of B episodes."""
        count = self.episodes_per_batch if episodes is None else episodes
        if count <= 0:
            raise ValueError("episodes must be > 0.")
        waves = (count + self.steps.num_envs - 1) // self.steps.num_envs
        return waves * self.steps.num_envs * self.horizon

    @torch.no_grad()
    def collect(
        self,
        *,
        policy_version: int,
        episodes: int | None = None,
        action_fn: Callable[[Observation], torch.Tensor] | None = None,
    ) -> EpisodeBatch:
        """Finish every selected episode before returning; policy never updated here."""
        remaining = self.episodes_per_batch if episodes is None else episodes
        if remaining <= 0:
            raise ValueError("episodes must be > 0.")
        started = perf_counter()
        waves = []
        masks = []
        interactions = 0
        reset_seconds = 0.0
        while remaining:
            count = min(remaining, self.steps.num_envs)
            reset_started = perf_counter()
            self.steps.env.reset()
            self.steps.reset_state(reset_environment=False)
            reset_seconds += perf_counter() - reset_started
            active = torch.arange(self.steps.num_envs, device=self.steps.device) < count
            records, valid = [], []
            for _ in range(self.horizon):
                valid.append(active.clone())
                step = self.steps.collect_step(action_fn)
                records.append(step)
                interactions += self.steps.num_envs
                active &= ~(step.terminated | step.truncated)
                if not bool(active.any()):
                    break
            if bool(active.any()):
                raise RuntimeError(
                    f"Episodes did not end within env.max_episode_length={self.horizon}. "
                    "Configure an explicit environment timeout; do not synthesize terminal flags."
                )
            rollout = self.steps.stack_steps(records, policy_version=policy_version)
            waves.append((rollout, count))
            mask = torch.stack(valid, dim=1)[:count]
            masks.append(
                torch.nn.functional.pad(mask, (0, self.horizon - mask.shape[1]))
            )
            remaining -= count

        def combine(values: list[Observation]) -> Observation:
            """Pad wave tensors [N,T,...] and concatenate selected episodes [B,H,...]."""
            if isinstance(values[0], dict):
                return {
                    key: combine([value[key] for value in values]) for key in values[0]
                }
            padded = []
            for value, (_, count) in zip(values, waves, strict=True):
                shape = (count, self.horizon, *value.shape[2:])
                target = value.new_zeros(shape)
                target[:, : value.shape[1]] = value[:count]
                padded.append(target)
            return torch.cat(padded, dim=0)

        values = {
            field.name: combine([getattr(wave, field.name) for wave, _ in waves])
            for field in fields(TransitionRollout)
            if field.name not in {"policy_version", "rollout_id"}
        }
        mask = torch.cat(masks, dim=0)
        # Mask trailing auto-reset episodes as padding, including all done flags.
        for key in values:
            values[key] = map_observation(
                values[key],
                lambda value: value.masked_fill(
                    ~mask.reshape(*mask.shape, *([1] * (value.ndim - 2))), 0
                ),
            )
        result = EpisodeBatch(
            TransitionRollout(
                **values, policy_version=policy_version, rollout_id=self._batch_id
            ),
            mask,
            interactions,
            perf_counter() - started,
            reset_seconds,
        )
        self._batch_id += 1
        return result

    def save_state(self) -> dict:
        return {"steps": self.steps.save_state(), "batch_id": self._batch_id}

    def load_state(self, state: dict) -> None:
        self.steps.load_state(state["steps"])
        self._batch_id = int(state["batch_id"])
