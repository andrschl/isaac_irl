from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from isaac_reward_learning.modules import RewardModel
from isaac_reward_learning.storage import ExpertRolloutStorage, ImitatorRolloutStorage

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

class IRL:
    rl_alg: PPO
    reward: RewardModel

    def __init__(
        self,
        rl_alg,
        reward,
        expert_data_path="",
        max_rollout_length=128,
        batch_size=256,
        num_learning_epochs=1,
        weight_decay=1e-6,
        max_grad_norm=1.0,
        reward_loss_coef=1.0,
        device="cpu",
    ):
        self.device = device

        # RL components
        self.rl_alg = rl_alg

        # IRL components
        self.reward = reward.to(self.device)
        self.expert_data_path = expert_data_path
        self.expert_storage = None  # initialized later
        self.imitator_storage = None  # initialized later
        self.transition = RolloutStorage.Transition()
        self.reward_optimizer = optim.RMSprop(self.reward.parameters(), lr=rl_alg.learning_rate, weight_decay=weight_decay)

        # IRL parameters 
        self.batch_size = batch_size
        self.num_learning_epochs = num_learning_epochs
        self.reward_loss_coef = reward_loss_coef
        self.max_grad_norm = max_grad_norm
        self.max_rollout_length = max_rollout_length

    def init_expert_storage(self, expert_data_path, obs_shape, actions_shape, **kwargs):
        self.expert_storage = ExpertRolloutStorage(
            expert_data_path, obs_shape, actions_shape, self.rl_alg.gamma, 
            max_traj_length=self.max_rollout_length, 
            device=self.device, **kwargs
        )

    def init_imitator_storage(self, num_envs, obs_shape, action_shape, **kwargs):
        self.imitator_storage = ImitatorRolloutStorage(
            num_envs, obs_shape, action_shape, self.rl_alg.gamma, 
            max_traj_length=self.max_rollout_length,
            device=self.device, **kwargs
        )

    def eval_mode(self):
        self.rl_alg.actor_critic.eval()
        self.reward.eval()

    def train_mode(self):
        self.rl_alg.actor_critic.train()
        self.reward.train()

    def act(self, obs):
        # Compute the actions and values
        self.transition.actions = self.rl_alg.actor_critic.act(obs).detach()
        self.transition.observations = obs
        return self.transition.actions
        

    def process_env_step(self, dones):
        self.transition.dones = dones
        self.imitator_storage.add_transition(self.transition)

    def reward_update(self):

        mean_reward_loss = 0
        expert_generator = self.expert_storage.mini_batch_generator(self.batch_size, 10**100)    # 10**100 is a large number of epochs
        imitator_generator = self.imitator_storage.mini_batch_generator(self.batch_size, self.num_learning_epochs)
        
        for i, (
            obs_batch,
            actions_batch,
            time_ids,
            num_imitator_samples
        ) in enumerate(imitator_generator):
           
            # Reward loss
            expert_obs_batch, expert_actions_batch, expert_time_ids, num_expert_samples = next(expert_generator)
            current_rewards = self.imitator_storage.get_expected_reward(
                self.reward, obs_batch, actions_batch, time_ids, num_imitator_samples
                )
            expert_rewards = self.expert_storage.get_expected_reward(
                self.reward, expert_obs_batch, expert_actions_batch, expert_time_ids, num_expert_samples
                )
            reward_loss = self.reward_loss_coef * (current_rewards - expert_rewards)

            # Gradient step
            self.reward_optimizer.zero_grad()
            reward_loss.backward()
            reward_gradient_norm = nn.utils.clip_grad_norm_(self.reward.parameters(), self.max_grad_norm)
            self.reward_optimizer.step()

            mean_reward_loss += reward_loss.item()

        num_updates = i
        mean_reward_loss /= num_updates

        return mean_reward_loss, reward_gradient_norm
