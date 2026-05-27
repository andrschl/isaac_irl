from __future__ import annotations

from typing import Any, Callable

import torch

from algorithms.bc import BC


class _RslRlActor:
    """Adapt an rsl_rl MLPModel to the core ``Actor`` protocol.

    `act` samples stochastically via the model's distribution head;
    `act_inference` returns the deterministic mean.
    """

    def __init__(self, mlp_actor: Any) -> None:
        self._actor = mlp_actor

    def act(self, obs: Any) -> torch.Tensor:
        return self._actor(obs, stochastic_output=True)

    def act_inference(self, obs: Any) -> torch.Tensor:
        return self._actor(obs)


class RslRlPpoAdapter:
    """Adapt rsl_rl PPO (and PPOWithBC) to the runner's ``RlAlgorithm`` API.

    The adapter owns the RL collection rollout. ``update()`` steps the env
    for ``steps_per_env_per_cycle`` ticks, scores each step with the learned
    reward model when configured, hands transitions to PPO via
    ``process_env_step``, then calls ``compute_returns`` and ``update``.
    BC-only mode (alpha >= 1) skips env stepping entirely.
    """

    def __init__(
        self,
        rl_alg: Any,
        *,
        env: Any,
        feature_map: Callable[[Any], torch.Tensor],
        reward_model: Callable[[torch.Tensor], torch.Tensor],
        steps_per_env_per_cycle: int,
        use_learned_reward: bool = True,
        device: str | torch.device = "cpu",
        bc_alg: BC | None = None,
    ) -> None:
        if steps_per_env_per_cycle <= 0:
            raise ValueError(
                f"`steps_per_env_per_cycle` must be > 0, got {steps_per_env_per_cycle}."
            )
        self._rl_alg = rl_alg
        self._env = env
        self._feature_map = feature_map
        self._reward_model = reward_model
        self._steps_per_env_per_cycle = int(steps_per_env_per_cycle)
        self._use_learned_reward = bool(use_learned_reward)
        self._device = torch.device(device)
        self._bc_alg = bc_alg
        self._last_bc_val_loss = float("nan")
        mlp_actor = getattr(rl_alg, "actor", None)
        if mlp_actor is None:
            raise AttributeError(
                "RslRlPpoAdapter requires `rl_alg.actor` (rsl_rl >= 4.0 MLPModel contract)."
            )
        self._actor_facade = _RslRlActor(mlp_actor)
        self._last_obs: Any = None

    @property
    def actor(self) -> _RslRlActor:
        """Side-effect-free policy facade for validation / imitator rollouts."""
        return self._actor_facade

    @property
    def _alpha(self) -> float:
        return 0.0 if self._bc_alg is None else float(self._bc_alg.cfg.alpha)

    @property
    def _is_bc_only(self) -> bool:
        return self._bc_alg is not None and self._alpha >= 1.0

    def update(self) -> dict[str, float]:
        if self._is_bc_only:
            result = self._rl_alg.bc_only_update()
        else:
            self._collect_rollout()
            result = self._rl_alg.update()

        metrics: dict[str, float] = {}
        if isinstance(result, dict):
            if "surrogate" in result:
                metrics["RL/policy_loss"] = float(result["surrogate"])
            if "bc" in result:
                metrics["BC/loss"] = float(result["bc"])
        else:
            metrics["RL/policy_loss"] = float("nan")
        return metrics

    def _collect_rollout(self) -> None:
        """Step the env for one cycle, scoring with the reward model and
        feeding rsl_rl PPO's per-transition recorder. Caches the trailing obs
        so the next cycle resumes without re-fetching from the env.
        """
        if self._last_obs is None:
            self._last_obs = self._obs_to_device(self._fetch_obs())
        obs = self._last_obs
        env_device = getattr(self._env, "device", self._device)
        num_envs = int(self._env.num_envs)
        with torch.inference_mode():
            for _ in range(self._steps_per_env_per_cycle):
                actions = self._rl_alg.act(obs)
                obs_next, env_rewards, dones, extras = self._env.step(actions.to(env_device))
                obs_next = self._obs_to_device(obs_next)
                env_rewards = env_rewards.to(self._device)
                done_vec = self._as_done_vector(dones, num_envs)
                if self._use_learned_reward:
                    features = self._features_for_step()
                    learned = self._reward_model(features).detach()
                    rewards = self._ensure_per_env(learned, num_envs).to(
                        device=env_rewards.device, dtype=env_rewards.dtype
                    )
                else:
                    rewards = env_rewards
                self._rl_alg.process_env_step(obs_next, rewards, done_vec, extras)
                obs = obs_next
        self._rl_alg.compute_returns(obs)
        self._last_obs = obs

    def _fetch_obs(self) -> Any:
        obs = self._env.get_observations()
        return obs[0] if isinstance(obs, tuple) else obs

    def _obs_to_device(self, obs: Any) -> Any:
        if isinstance(obs, torch.Tensor):
            return obs.to(self._device)
        to_device = getattr(obs, "to", None)
        if callable(to_device):
            return to_device(self._device)
        if isinstance(obs, dict):
            return {k: self._obs_to_device(v) for k, v in obs.items()}
        if isinstance(obs, tuple):
            return tuple(self._obs_to_device(v) for v in obs)
        if isinstance(obs, list):
            return [self._obs_to_device(v) for v in obs]
        return obs

    @staticmethod
    def _as_done_vector(dones: torch.Tensor, num_envs: int) -> torch.Tensor:
        if not isinstance(dones, torch.Tensor):
            dones = torch.as_tensor(dones)
        done_vec = dones.reshape(-1).to(torch.bool)
        if tuple(done_vec.shape) != (num_envs,):
            raise ValueError(
                f"Expected dones shape ({num_envs},), got {tuple(dones.shape)}."
            )
        return done_vec

    def _features_for_step(self) -> torch.Tensor:
        features = self._feature_map(self._env)
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)
        features = features.to(self._device)
        if features.ndim != 2:
            raise ValueError(
                f"Expected feature_map(env) shape [N, D], got {tuple(features.shape)}."
            )
        return features

    @staticmethod
    def _ensure_per_env(rewards: torch.Tensor, num_envs: int) -> torch.Tensor:
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.as_tensor(rewards)
        if rewards.ndim == 2 and rewards.shape[-1] == 1:
            rewards = rewards.squeeze(-1)
        if rewards.ndim != 1 or rewards.shape[0] != num_envs:
            raise ValueError(
                f"Expected learned reward shape [{num_envs}], got {tuple(rewards.shape)}. "
                "Reward model must produce one scalar reward per env."
            )
        return rewards

    def train_metrics(self) -> dict[str, float]:
        metrics = self._policy_std_stats()
        if self._bc_alg is not None:
            metrics["BC/alpha"] = self._alpha
        return metrics

    def eval_metrics(self) -> dict[str, float]:
        if self._bc_alg is None:
            return {}
        actor = getattr(self._rl_alg, "actor", None)
        if actor is None:
            return {}
        val_loss = self._bc_alg.compute_validation_loss(actor)
        if val_loss is None:
            return {}
        self._last_bc_val_loss = float(val_loss)
        return {"BC/val_loss": self._last_bc_val_loss}

    def train_mode(self) -> None:
        self._rl_alg.train_mode()

    def eval_mode(self) -> None:
        self._rl_alg.eval_mode()

    def save_state(self) -> dict[str, Any]:
        return dict(self._rl_alg.save())

    def load_state(self, checkpoint: dict[str, Any], *, load_optimizer: bool = True) -> None:
        load_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": bool(load_optimizer),
            "iteration": True,
            "rnd": True,
        }
        self._rl_alg.load(checkpoint, load_cfg=load_cfg, strict=True)

    def _policy_std_stats(self) -> dict[str, float]:
        actor = getattr(self._rl_alg, "actor", None)
        dist = getattr(actor, "distribution", None) if actor is not None else None
        if dist is None:
            return {}
        with torch.no_grad():
            if hasattr(dist, "std_param"):
                std = dist.std_param.detach()
            elif hasattr(dist, "log_std_param"):
                std = dist.log_std_param.detach().exp()
            else:
                return {}
        return {
            "RL/policy_std_min": float(std.min().item()),
            "RL/policy_std_max": float(std.max().item()),
            "RL/policy_std_mean": float(std.mean().item()),
        }
