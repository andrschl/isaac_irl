from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

from algorithms import IRL
from utils.runtime_context import RuntimeContext
from storage.feature_storage import FeatureBufCfg


@dataclass(slots=True)
class IrlRunnerCfg:
    """Runner/trainer-only config."""

    num_steps_per_env_rl: int = 24
    save_interval: int = 50

    policy_updates_per_cycle: int = 1
    reward_updates_per_cycle: int = 1

    imitator_buffer: FeatureBufCfg = field(default_factory=FeatureBufCfg)
    expert_buffer: FeatureBufCfg = field(default_factory=lambda: FeatureBufCfg(min_ep_len=1))
    expert_num_envs: int = 1


ExpertBufferLoader = Callable[[Any], None]
RewardEnvWrapperFactory = Callable[[Any, torch.nn.Module, Callable[[Any], torch.Tensor]], Any]


class IrlRunner:
    """On-policy PPO + feature-buffer IRL runner."""

    def __init__(
        self,
        env,
        *,
        rl_alg,
        irl_alg: IRL,
        feature_map: Callable[[Any], torch.Tensor],
        runner_cfg: IrlRunnerCfg,
        log_dir: str | None = None,
        device: str | torch.device = "cpu",
        runtime_ctx: RuntimeContext | None = None,
        expert_buffer_loader: ExpertBufferLoader | None = None,
        reward_env_wrapper_factory: RewardEnvWrapperFactory | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.base_env = env
        self.env = env
        self.rl_alg = rl_alg
        self.irl_alg = irl_alg
        self.feature_map = feature_map
        self.cfg = runner_cfg
        self.log_dir = log_dir
        self.writer: TensorboardSummaryWriter | None = None
        self.current_learning_iteration = 0

        self.num_steps_per_env_rl = int(runner_cfg.num_steps_per_env_rl)
        self.save_interval = int(runner_cfg.save_interval)
        self.policy_updates_per_cycle = int(runner_cfg.policy_updates_per_cycle)
        self.reward_updates_per_cycle = int(runner_cfg.reward_updates_per_cycle)
        self.expert_num_envs = int(runner_cfg.expert_num_envs)
        self.imitator_buffer_cfg = runner_cfg.imitator_buffer
        self.expert_buffer_cfg = runner_cfg.expert_buffer

        # Optional learned-reward wrapper
        if reward_env_wrapper_factory is not None:
            self.env = reward_env_wrapper_factory(self.base_env, self.irl_alg.reward, self.feature_map)
            self.irl_alg.env = self.env
        else:
            self.irl_alg.env = self.base_env

        obs0 = self._get_obs().to(self.device)
        self.rl_alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env_rl,
            obs0,
            [self.env.num_actions],
        )

        if runtime_ctx is None:
            runtime_ctx = RuntimeContext(
                num_envs=int(self.env.num_envs),
                feature_dim=int(self.feature_map(self.env).shape[1]),
                device=str(self.device),
            )

        self.irl_alg.init_imitator_storage(runtime_ctx=runtime_ctx, cfg=self.imitator_buffer_cfg)
        self.irl_alg.init_expert_storage(
            runtime_ctx=runtime_ctx,
            cfg=self.expert_buffer_cfg,
            num_envs=self.expert_num_envs,
        )

        if expert_buffer_loader is not None:
            expert_buffer_loader(self.irl_alg.expert_storage)

        self._last_reward_loss = float("nan")
        self._last_reward_grad_norm = float("nan")

    def _get_obs(self):
        obs = self.env.get_observations()
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def _ensure_writer(self) -> None:
        if self.log_dir is None or self.writer is not None:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)

    def _collect_imitator_features(self, dones: torch.Tensor) -> None:
        if self.irl_alg.imitator_storage is None:
            return
        features = self.feature_map(self.env)
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features)
        self.irl_alg.process_env_step(dones=dones, features=features.to(self.device))

    def _collect_rollout_batch(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            for _ in range(self.num_steps_per_env_rl):
                actions = self.rl_alg.act(obs)
                obs_next, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                obs_next = obs_next.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)

                self.rl_alg.process_env_step(obs_next, rewards, dones, extras)
                self._collect_imitator_features(dones=dones.to(torch.bool))
                obs = obs_next

            self.rl_alg.compute_returns(obs)
        return obs

    def _run_policy_updates(self) -> None:
        for _ in range(self.policy_updates_per_cycle):
            self.rl_alg.update()

    def _has_reward_data(self) -> bool:
        expert_storage = self.irl_alg.expert_storage
        imitator_storage = self.irl_alg.imitator_storage
        if expert_storage is None or imitator_storage is None:
            return False
        return len(expert_storage) > 0 and len(imitator_storage) > 0

    def _run_reward_updates(self) -> None:
        if not self._has_reward_data():
            return
        for _ in range(self.reward_updates_per_cycle):
            self._last_reward_loss, self._last_reward_grad_norm = self.irl_alg.reward_update()

    def _log_iteration(self, iteration: int) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar("IRL/reward_loss", self._last_reward_loss, iteration)
        self.writer.add_scalar("IRL/reward_grad_norm", self._last_reward_grad_norm, iteration)

    def _maybe_save_checkpoint(self, iteration: int) -> None:
        if self.log_dir is None:
            return
        if (iteration + 1) % self.save_interval != 0:
            return
        self.save(os.path.join(self.log_dir, f"model_{iteration + 1}.pt"))

    def train_mode(self) -> None:
        policy = getattr(self.rl_alg, "policy", None)
        if policy is not None:
            policy.train()
        self.irl_alg.reward.train()

    def eval_mode(self) -> None:
        policy = getattr(self.rl_alg, "policy", None)
        if policy is not None:
            policy.eval()
        self.irl_alg.reward.eval()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self._ensure_writer()


        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf") and hasattr(self.env, "max_episode_length"):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self._get_obs().to(self.device)
        self.train_mode()

        start_iter = self.current_learning_iteration
        for it in range(start_iter, start_iter + int(num_learning_iterations)):
            self.irl_alg.clear_imitator_storage()
            obs = self._collect_rollout_batch(obs)
            self._run_policy_updates()
            self._run_reward_updates()
            self.current_learning_iteration = it + 1
            self._log_iteration(it)
            self._maybe_save_checkpoint(it)

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        policy = getattr(self.rl_alg, "policy", None)
        payload = {
            "iter": int(self.current_learning_iteration),
        }
        if policy is not None:
            payload["model_state_dict"] = policy.state_dict()
        if hasattr(self.rl_alg, "optimizer") and self.rl_alg.optimizer is not None:
            payload["optimizer_state_dict"] = self.rl_alg.optimizer.state_dict()
        payload["reward_model_state_dict"] = self.irl_alg.reward.state_dict()
        payload["reward_optimizer_state_dict"] = self.irl_alg.reward_optimizer.state_dict()
        torch.save(payload, path)

    def load(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location=self.device)

        model_state = checkpoint.get("model_state_dict")
        policy = getattr(self.rl_alg, "policy", None)
        if model_state is not None and policy is not None:
            policy.load_state_dict(model_state)

        if load_optimizer and hasattr(self.rl_alg, "optimizer") and self.rl_alg.optimizer is not None:
            optim_state = checkpoint.get("optimizer_state_dict")
            if optim_state is not None:
                self.rl_alg.optimizer.load_state_dict(optim_state)

        reward_state = checkpoint.get("reward_model_state_dict")
        if reward_state is not None:
            self.irl_alg.reward.load_state_dict(reward_state)

        reward_optim_state = checkpoint.get("reward_optimizer_state_dict")
        if load_optimizer and reward_optim_state is not None:
            self.irl_alg.reward_optimizer.load_state_dict(reward_optim_state)

        self.current_learning_iteration = int(checkpoint.get("iter", 0))
        return checkpoint.get("infos", None)
