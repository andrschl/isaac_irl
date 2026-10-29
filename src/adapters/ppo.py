"""Adapt rsl_rl PPO (and PPOWithBC) to the runner's algorithm contract."""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange

from algorithms.bc import BC
from interfaces import FeatureSource, Generation
from reward.scorer import RewardScorer
from storage.observations import Observation, detached_observation


class _RslRlActor:
    """Adapt an rsl_rl MLPModel to the core ``Actor`` protocol.

    `act` samples stochastically via the model's distribution head;
    `act_inference` returns the deterministic mean.
    """

    def __init__(self, mlp_actor: Any) -> None:
        self._actor = mlp_actor

    def act(self, obs: Observation) -> torch.Tensor:
        return self._actor(obs, stochastic_output=True)

    def act_inference(self, obs: Observation) -> torch.Tensor:
        return self._actor(obs)


class RslRlPpoAdapter:
    """Collect one on-policy rollout, then run one PPO update over it.

    ``reward`` decides what the policy maximizes: a scorer scores the reward
    features of every step, and ``None`` passes the environment's own reward
    through unchanged. ``feature_source`` is what the reward learner consumes;
    it is required whenever a scorer is present, because the same features feed
    both.
    """

    def __init__(
        self,
        rl_alg: Any,
        *,
        env: Any,
        feature_source: FeatureSource | None,
        reward: RewardScorer | None,
        steps_per_env_per_cycle: int,
        entropy_rate: float = 0.0,
        step_dt: float = 1.0,
        device: str | torch.device = "cpu",
        bc_alg: BC | None = None,
    ) -> None:
        if steps_per_env_per_cycle <= 0:
            raise ValueError(
                f"`steps_per_env_per_cycle` must be > 0, got {steps_per_env_per_cycle}."
            )
        if reward is not None and feature_source is None:
            raise ValueError(
                "A learned reward scores reward features, so `feature_source` is "
                "required whenever `reward` is given."
            )
        if entropy_rate < 0.0:
            raise ValueError(f"entropy_rate must be >= 0, got {entropy_rate}.")
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be > 0, got {step_dt}.")
        if not hasattr(rl_alg, "actor"):
            raise TypeError(
                "RslRlPpoAdapter requires `rl_alg.actor` "
                f"(rsl_rl >= 4.0 MLPModel contract); got {type(rl_alg).__name__}."
            )
        self._rl_alg = rl_alg
        self._env = env
        self._feature_source = feature_source
        self._reward = reward
        self._steps_per_env_per_cycle = int(steps_per_env_per_cycle)
        self._entropy_rate = float(entropy_rate)
        self._step_dt = float(step_dt)
        self._device = torch.device(device)
        self._bc_alg = bc_alg
        self._actor_module = rl_alg.actor
        self._actor_facade = _RslRlActor(self._actor_module)
        self._observations: Observation | None = None
        self._collection_metrics: dict[str, float] = {}

    @property
    def actor(self) -> _RslRlActor:
        """Side-effect-free policy facade for validation / video rollouts."""
        return self._actor_facade

    @property
    def simulator_steps_per_update(self) -> int:
        return self._steps_per_env_per_cycle * int(self._env.num_envs)

    @property
    def _bc_weight(self) -> float:
        return 0.0 if self._bc_alg is None else float(self._bc_alg.cfg.weight)

    @property
    def _is_bc_only(self) -> bool:
        return self._bc_alg is not None and self._bc_weight >= 1.0

    def reset_collection_state(self) -> None:
        """Drop cached observations after an external rollout moved the env."""
        self._observations = None

    def collect(self) -> Generation:
        """Step the env for one cycle and hand PPO every transition.

        Shapes:
            features: [N, T, D] when a feature source is configured
        """
        if self._is_bc_only:
            return Generation(transition_count=0, features=None)

        observations = self._current_observations()
        env_device = self._env.device
        num_envs = int(self._env.num_envs)
        step_features: list[torch.Tensor] = []
        step_dones: list[torch.Tensor] = []
        environment_reward_sum = torch.zeros((), device=self._device)
        entropy_bonus_sum = torch.zeros((), device=self._device)
        # Isaac may lazily cache simulator-state tensors during env.step().
        # no_grad avoids autograd without creating immutable inference tensors.
        with torch.no_grad():
            for _ in range(self._steps_per_env_per_cycle):
                actions = self._rl_alg.act(observations)
                entropy_bonus = self._sampled_entropy_bonus(num_envs)
                next_observations, environment_rewards, dones, extras = self._env.step(
                    actions.to(env_device)
                )
                next_observations = detached_observation(
                    next_observations, self._device
                )
                environment_rewards = environment_rewards.to(self._device)
                environment_reward_sum += environment_rewards.sum()
                features = self._step_features(num_envs)
                if features is not None:
                    step_features.append(features)
                task_rewards = self._step_rewards(
                    features, environment_rewards, num_envs
                )
                rewards = task_rewards + entropy_bonus.to(task_rewards)
                entropy_bonus_sum += entropy_bonus.sum()
                done_vector = self._done_vector(dones, num_envs)
                step_dones.append(done_vector)
                self._rl_alg.process_env_step(
                    next_observations, rewards, done_vector, extras
                )
                observations = next_observations
        self._observations = observations
        self._rl_alg.compute_returns(observations)
        dones = torch.stack(step_dones, dim=1)  # [N,T]
        transitions = self._steps_per_env_per_cycle * num_envs
        self._collection_metrics = {
            "Train/reward_per_step": float(environment_reward_sum / transitions),
            "Train/entropy_bonus_per_step": float(entropy_bonus_sum / transitions),
            "RL/entropy_rate": self._entropy_rate,
            "RL/entropy_step_coefficient": self._step_dt * self._entropy_rate,
        }
        return Generation(
            transition_count=transitions,
            features=(
                rearrange(torch.stack(step_features, dim=0), "t n d -> n t d")
                if step_features
                else None
            ),
            dones=dones,
            completed_episodes=int(dones.sum()),
        )

    def update(self) -> dict[str, float]:
        """Run one PPO (or BC-only) update over the collected rollout."""
        result = (
            self._rl_alg.bc_only_update() if self._is_bc_only else self._rl_alg.update()
        )
        if not isinstance(result, dict) or "surrogate" not in result:
            raise TypeError(
                "Expected the rsl_rl update to return a dict containing "
                f"'surrogate', got {result!r}."
            )
        metrics = {"RL/policy_loss": float(result["surrogate"])}
        if "value" in result:
            metrics["RL/value_loss"] = float(result["value"])
        elif "value_function" in result:
            metrics["RL/value_loss"] = float(result["value_function"])
        if "bc" in result:
            metrics["BC/loss"] = float(result["bc"])
        if "entropy" in result:
            metrics["RL/policy_entropy"] = float(result["entropy"])
        return metrics

    def _sampled_entropy_bonus(self, num_envs: int) -> torch.Tensor:
        """Return ``-dt * alpha * log pi(a|s)`` for sampled actions [N]."""
        if self._entropy_rate == 0.0:
            return torch.zeros(num_envs, device=self._device)
        transition = getattr(self._rl_alg, "transition", None)
        log_probability = getattr(transition, "actions_log_prob", None)
        if log_probability is None:
            raise AttributeError(
                "Entropy-regularized PPO requires the backend transition's "
                "sampled actions_log_prob after act()."
            )
        log_probability = log_probability.reshape(-1).to(self._device)
        if tuple(log_probability.shape) != (num_envs,):
            raise ValueError(
                f"Expected sampled action log probabilities [{num_envs}], got "
                f"{tuple(log_probability.shape)}."
            )
        if not torch.isfinite(log_probability).all():
            raise FloatingPointError("PPO sampled non-finite action log probabilities.")
        return -self._step_dt * self._entropy_rate * log_probability

    def _current_observations(self) -> Observation:
        if self._observations is None:
            observations = self._env.get_observations()
            self._observations = detached_observation(observations, self._device)
        return self._observations

    def _step_features(self, num_envs: int) -> torch.Tensor | None:
        """Reward features of the step just taken, shaped [N, D]."""
        if self._feature_source is None:
            return None
        features = self._feature_source.features(self._env).to(self._device)
        expected = (num_envs, int(self._feature_source.spec.dim))
        if tuple(features.shape) != expected:
            raise ValueError(
                f"Expected features {expected}, got {tuple(features.shape)}."
            )
        return features

    def _step_rewards(
        self,
        features: torch.Tensor | None,
        environment_rewards: torch.Tensor,
        num_envs: int,
    ) -> torch.Tensor:
        if self._reward is None:
            return environment_rewards
        if features is None:
            raise RuntimeError("A learned reward requires per-step features.")
        rewards = self._reward.score(features).detach()
        if tuple(rewards.shape) != (num_envs,):
            raise ValueError(
                f"Expected learned reward [{num_envs}], got {tuple(rewards.shape)}. "
                "The reward model must produce one scalar reward per env."
            )
        return rewards.to(
            device=environment_rewards.device, dtype=environment_rewards.dtype
        )

    @staticmethod
    def _done_vector(dones: torch.Tensor, num_envs: int) -> torch.Tensor:
        done_vector = dones.reshape(-1).to(torch.bool)
        if tuple(done_vector.shape) != (num_envs,):
            raise ValueError(
                f"Expected dones shape ({num_envs},), got {tuple(dones.shape)}."
            )
        return done_vector

    def train_metrics(self) -> dict[str, float]:
        metrics = {**self._policy_std_metrics(), **self._collection_metrics}
        reward_manager = (
            getattr(self._env.unwrapped, "reward_manager", None)
            if hasattr(self._env, "unwrapped")
            else None
        )
        if reward_manager is not None:
            metrics.update(
                {
                    f"RewardWeights/{name}": float(
                        reward_manager.get_term_cfg(name).weight
                    )
                    for name in reward_manager.active_terms
                }
            )
        if self._bc_alg is not None:
            metrics["BC/weight"] = self._bc_weight
        return metrics

    def eval_metrics(self) -> dict[str, float]:
        if self._bc_alg is None:
            return {}
        validation_loss = self._bc_alg.compute_validation_loss(self._actor_module)
        if validation_loss is None:
            return {}
        return {"BC/val_loss": float(validation_loss)}

    def train_mode(self) -> None:
        self._rl_alg.train_mode()

    def eval_mode(self) -> None:
        self._rl_alg.eval_mode()

    def save_state(self) -> dict[str, Any]:
        return dict(self._rl_alg.save())

    def load_state(
        self, checkpoint: dict[str, Any], *, load_optimizer: bool = True
    ) -> None:
        self._rl_alg.load(
            checkpoint,
            load_cfg={
                "actor": True,
                "critic": True,
                "optimizer": bool(load_optimizer),
                "iteration": True,
                "rnd": True,
            },
            strict=True,
        )

    def _policy_std_metrics(self) -> dict[str, float]:
        """Report the Gaussian head's action standard deviation."""
        distribution = getattr(self._actor_module, "distribution", None)
        if distribution is None:
            return {}
        with torch.no_grad():
            if hasattr(distribution, "std_param"):
                std = distribution.std_param.detach()
            elif hasattr(distribution, "log_std_param"):
                std = distribution.log_std_param.detach().exp()
            else:
                raise AttributeError(
                    "Expected the actor distribution to expose `std_param` or "
                    f"`log_std_param`, got {type(distribution).__name__}."
                )
        return {
            "RL/policy_std_min": float(std.min().item()),
            "RL/policy_std_max": float(std.max().item()),
            "RL/policy_std_mean": float(std.mean().item()),
        }
