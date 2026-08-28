"""Fixed-reward rsl-rl PPO adapter used by the sample-efficiency benchmark."""

from __future__ import annotations

from typing import Any

import torch

from adapters.action_diagnostics import clipping_metrics


class FixedRewardPpoAdapter:
    """Collect one RSL-RL PPO rollout against the immutable environment reward."""

    def __init__(
        self,
        *,
        env: Any,
        algorithm: Any,
        steps_per_env: int,
        device: str | torch.device,
    ) -> None:
        self.env = env
        self.algorithm = algorithm
        self.steps_per_env = int(steps_per_env)
        self.device = torch.device(device)
        self.num_envs = int(env.num_envs)
        self._actor = _PpoActor(algorithm.actor)

    @property
    def actor(self) -> "_PpoActor":
        return self._actor

    @property
    def simulator_steps_per_update(self) -> int:
        return self.num_envs * self.steps_per_env

    def update(self) -> dict[str, float]:
        observations = self._get_observations()
        rewards_sum = torch.zeros(self.num_envs, device=self.device)
        done_count = 0
        raw_action_steps: list[torch.Tensor] = []
        executed_action_steps: list[torch.Tensor] = []
        # None means the environment executes what the policy asked for; the
        # surrogate then optimizes the action that was actually taken.
        environment_clip = getattr(self.env, "clip_actions", None)
        with torch.no_grad():
            for _ in range(self.steps_per_env):
                actions = self.algorithm.act(observations)
                executed_actions = (
                    actions
                    if environment_clip is None
                    else actions.clamp(
                        -float(environment_clip), float(environment_clip)
                    )
                )
                raw_action_steps.append(actions.detach())
                executed_action_steps.append(executed_actions.detach())
                next_observations, rewards, done, extras = self.env.step(
                    executed_actions.to(self.env.device)
                )
                reward_tensor = torch.as_tensor(rewards, device=self.device).reshape(-1)
                done_tensor = (
                    torch.as_tensor(done, device=self.device).reshape(-1).bool()
                )
                self.algorithm.process_env_step(
                    next_observations,
                    reward_tensor,
                    done_tensor,
                    extras,
                )
                observations = next_observations
                rewards_sum += reward_tensor
                done_count += int(done_tensor.sum().item())
            self.algorithm.compute_returns(observations)
        losses = self.algorithm.update()
        metrics = {
            "PPO/policy_loss": float(losses["surrogate"]),
            "PPO/value_loss": float(losses["value"]),
            "PPO/entropy": float(losses["entropy"]),
            "PPO/learning_rate": float(self.algorithm.learning_rate),
            "Train/reward_per_env_step": float(
                rewards_sum.mean().item() / self.steps_per_env
            ),
            "Train/episodes_completed": float(done_count),
        }
        raw_actions = torch.stack(raw_action_steps)
        metrics.update(
            clipping_metrics(
                raw_actions,
                torch.stack(executed_action_steps),
                bound=(
                    float(environment_clip)
                    if environment_clip is not None
                    else float(raw_actions.abs().max().item()) or 1.0
                ),
            )
        )
        distribution = self.algorithm.actor.distribution
        if hasattr(distribution, "std_param"):
            std = distribution.std_param.detach()
        else:
            std = distribution.log_std_param.detach().exp()
        metrics["RL/policy_std_mean"] = float(std.mean().item())
        return metrics

    def _get_observations(self) -> Any:
        observations = self.env.get_observations()
        return observations[0] if isinstance(observations, tuple) else observations

    def reset_collection_state(self) -> None:
        self.env.reset()

    def train_metrics(self) -> dict[str, float]:
        return {}

    def eval_metrics(self) -> dict[str, float]:
        return {}

    def train_mode(self) -> None:
        self.algorithm.train_mode()

    def eval_mode(self) -> None:
        self.algorithm.eval_mode()

    def save_state(self) -> dict[str, Any]:
        return dict(self.algorithm.save())

    def load_state(
        self, checkpoint: dict[str, Any], *, load_optimizer: bool = True
    ) -> None:
        self.algorithm.load(
            checkpoint,
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": load_optimizer,
                "iteration": True,
                "rnd": True,
            },
            strict=True,
        )


class _PpoActor:
    """Side-effect-free RSL actor facade."""

    def __init__(self, actor: Any) -> None:
        self.actor = actor

    def act(self, observations: Any) -> torch.Tensor:
        return self.actor(observations, stochastic_output=True)

    def act_inference(self, observations: Any) -> torch.Tensor:
        return self.actor(observations)
