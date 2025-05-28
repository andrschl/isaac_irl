from typing import Any, Callable, Mapping, Optional, Tuple, Union

import copy
import itertools
import math
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.agents.torch.ppo import PPO

import matplotlib.pyplot as plt
import numpy as np


# fmt: off
# [start-config-dict-torch]
GAIL_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 6,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch
    "n_discriminator_updates": 1,   # number of discriminator updates during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 5e-5,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})
    "amp_state_preprocessor": None,         # AMP state preprocessor class (see skrl.resources.preprocessors)
    "amp_state_preprocessor_kwargs": {},    # AMP state preprocessor's kwargs (e.g. {"size": env.amp_observation_space})
    "amp_action_preprocessor": None,
    "amp_action_preprocessor_kwargs": {},

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.0,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,          # entropy loss scaling factor
    "value_loss_scale": 2.5,            # value loss scaling factor
    "discriminator_loss_scale": 5.0,    # discriminator loss scaling factor

    "amp_batch_size": 512,                  # batch size for updating the reference motion dataset
    "discriminator_batch_size": 0,          # batch size for computing the discriminator loss (all samples if 0)
    "discriminator_reward_scale": 2,                    # discriminator reward scaling factor
    "discriminator_logit_regularization_scale": 0.05,   # logit regularization scale factor for the discriminator loss
    "discriminator_gradient_penalty_scale": 5,          # gradient penalty scaling factor for the discriminator loss
    "discriminator_weight_decay_scale": 0.0001,         # weight decay scaling factor for the discriminator loss

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


def mixup_state_action(self, x_s_expert, x_a_expert, x_s_agent, x_a_agent, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x_s_expert.size()[0]
    index = torch.randperm(batch_size).to(self.device)
    mixed_x_s = lam * x_s_expert + (1 - lam) * x_s_agent[index, :]
    mixed_x_a = lam * x_a_expert + (1 - lam) * x_a_agent[index, :]

    return mixed_x_s, mixed_x_a, lam


class GAIL(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Memory,
        observation_space: Union[int, Tuple[int], gymnasium.Space],
        action_space: Union[int, Tuple[int], gymnasium.Space],
        device: Union[str, torch.device],
        cfg: dict,
        amp_observation_space: Union[int, Tuple[int], gymnasium.Space],
        motion_dataset: Memory,
        reply_buffer: Memory,
        collect_reference_motions: Callable[[int], tuple[torch.Tensor, torch.Tensor]] = None,
        collect_observation: Optional[Callable[[], torch.Tensor]] = None,
    ) -> None:
        """Adversarial Motion Priors (AMP)

        https://arxiv.org/abs/2104.02180

        The implementation is adapted from the NVIDIA IsaacGymEnvs
        (https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/learning/amp_continuous.py)

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        :param amp_observation_space: AMP observation/state space or shape (default: ``None``)
        :type amp_observation_space: int, tuple or list of int, gymnasium.Space or None
        :param motion_dataset: Reference motion dataset: M (default: ``None``)
        :type motion_dataset: skrl.memory.torch.Memory or None
        :param reply_buffer: Reply buffer for preventing discriminator overfitting: B (default: ``None``)
        :type reply_buffer: skrl.memory.torch.Memory or None
        :param collect_reference_motions: Callable to collect reference motions (default: ``None``)
        :type collect_reference_motions: Callable[[int], torch.Tensor] or None
        :param collect_observation: Callable to collect observation (default: ``None``)
        :type collect_observation: Callable[[], torch.Tensor] or None

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(GAIL_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )
        self.first_run = True
        
        self.observation_space: Union[int, Tuple[int], gymnasium.Space]
        self.action_space: Union[int, Tuple[int], gymnasium.Space]

        self.amp_observation_space = amp_observation_space
        self.motion_dataset = motion_dataset
        self.reply_buffer = reply_buffer
        self.collect_reference_motions = collect_reference_motions
        self.collect_observation = collect_observation

        # models
        self.policy: Model = self.models.get("policy", None)
        self.value: Model = self.models.get("value", None)
        self.discriminator: Model = self.models.get("discriminator", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value
        self.checkpoint_modules["discriminator"] = self.discriminator

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.value is not None:
                self.value.broadcast_parameters()
            if self.discriminator is not None:
                self.discriminator.broadcast_parameters()

        # PLotting
        plt.ion()
        self.fig, self.ax = plt.subplots(5, 1, figsize=(10, 10))
        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self.n_discriminator_updates = self.cfg["n_discriminator_updates"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]
        self._discriminator_loss_scale = self.cfg["discriminator_loss_scale"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]
        self._amp_state_preprocessor = self.cfg["amp_state_preprocessor"]
        self._amp_action_preprocessor = self.cfg["amp_action_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._amp_batch_size = self.cfg["amp_batch_size"]

        self._discriminator_batch_size = self.cfg["discriminator_batch_size"]
        self._discriminator_reward_scale = self.cfg["discriminator_reward_scale"]
        self._discriminator_logit_regularization_scale = self.cfg["discriminator_logit_regularization_scale"]
        self._discriminator_gradient_penalty_scale = self.cfg["discriminator_gradient_penalty_scale"]
        self._discriminator_weight_decay_scale = self.cfg["discriminator_weight_decay_scale"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
            self.scaler_disc = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)
            self.scaler_disc = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None and self.discriminator is not None:
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.policy.parameters(), self.value.parameters()),
                lr=self._learning_rate,
            )
            self.optimizer_disc = torch.optim.Adam(
                itertools.chain(self.discriminator.parameters()),
                lr=self._learning_rate,
            )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.scheduler_disc = self._learning_rate_scheduler(
                    self.optimizer_disc, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["optimizer"] = self.optimizer
            self.checkpoint_modules["optimizer_disc"] = self.optimizer_disc

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

        if self._amp_state_preprocessor:
            self._amp_state_preprocessor = self._amp_state_preprocessor(**self.cfg["amp_state_preprocessor_kwargs"])
            self.checkpoint_modules["amp_state_preprocessor"] = self._amp_state_preprocessor
        else:
            self._amp_state_preprocessor = self._empty_preprocessor

        if self._amp_action_preprocessor:
            self._amp_action_preprocessor = self._amp_action_preprocessor(**self.cfg["amp_action_preprocessor_kwargs"])
            self.checkpoint_modules["amp_action_preprocessor"] = self._amp_action_preprocessor
        else:
            self._amp_action_preprocessor = self._empty_preprocessor
            

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory: Memory
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="next_values", size=1, dtype=torch.float32)

        self.tensors_names = [
            "states",
            "actions",
            "rewards",
            "next_states",
            "terminated",
            "log_prob",
            "values",
            "returns",
            "advantages",
            "next_values",
        ]

        # create tensors for motion dataset and reply buffer
        if self.motion_dataset is not None:
            self.motion_dataset.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)
            self.motion_dataset.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.reply_buffer.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)
            self.reply_buffer.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            for i in range(math.ceil(self.motion_dataset.memory_size / self._amp_batch_size)):
                if i % 10000 == 0:
                    print(f"Collecting reference motions {self.motion_dataset.memory_index}")
                motions = self.collect_reference_motions(self._amp_batch_size)
                self.motion_dataset.add_samples(states=motions[0][0], actions=motions[0][1])
                
        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> tuple:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # use collected states
        if self._current_states is not None:
            states = self._current_states
        states = self._state_preprocessor(states)
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": states}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_prob, outputs = self.policy.act({"states": states}, role="policy")
            self._current_log_prob = log_prob
        return actions, log_prob, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # use collected states
        if self._current_states is not None:
            states = self._current_states

        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
                values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # compute next values
            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                next_values, _, _ = self.value.act({"states": self._state_preprocessor(next_states)}, role="value")
                next_values = self._value_preprocessor(next_values, inverse=True)
                if "terminate" in infos:
                    next_values *= infos["terminate"].view(-1, 1).logical_not()  # compatibility with IsaacGymEnvs
                else:
                    next_values *= terminated.view(-1, 1).logical_not()

            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
                next_values=next_values,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    log_prob=self._current_log_prob,
                    values=values,
                    next_values=next_values,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.collect_observation is not None:
            self._current_states = self.collect_observation()

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self._update_discriminator(timestep, timesteps)
            self.set_mode("eval")

        super().post_interaction(timestep, timesteps) # write tracking data and checkpoints

        
    def _update_discriminator(self, timestep: int, timesteps: int) -> None:
        for i in range(self.n_discriminator_updates):
            # print(f"[DEBUG] {self.reply_buffer.memory_index}")
            sampled_replay_batches = self.reply_buffer.sample(
                names=["states", "actions"],
                batch_size=self.memory.num_envs,
                mini_batches=self._mini_batches,
                sequence_length=self._amp_batch_size,
            )
            sampled_expert_batches = self.motion_dataset.sample(
                names=["states", "actions"],
                batch_size=self.memory.num_envs,
                mini_batches=self._mini_batches,
                sequence_length=self._amp_batch_size,
            )
            
            cumulative_discriminator_loss = 0
            for batch_index, (generator_sample, expert_sample) in enumerate(zip(sampled_replay_batches, sampled_expert_batches)):
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):                
                    # Get from replay buffer
                    sampled_generator_states = self._amp_state_preprocessor(generator_sample[0], train=True)
                    sampled_generator_actions = self._amp_action_preprocessor(generator_sample[1], train=True)
                    # Get from motion dataset   
                    sampled_expert_states = self._amp_state_preprocessor(expert_sample[0], train=True)
                    sampled_expert_actions = self._amp_action_preprocessor(expert_sample[1], train=True)                  

                    if i == 0 and (timestep-99) % 1000 == 0 and False:
                        plot_trajectories(
                            sampled_generator_states, sampled_generator_actions, 
                            sampled_expert_states, sampled_expert_actions,
                            self.fig, self.ax
                        )
                            
                    sampled_expert_states.requires_grad_(True)
                    sampled_expert_actions.requires_grad_(True)
                    generator_logits, _, _ = self.discriminator.act({"states": sampled_generator_states, "taken_actions": sampled_generator_actions}, role="discriminator")
                    expert_logits, _, _ = self.discriminator.act({"states": sampled_expert_states, "taken_actions": sampled_expert_actions}, role="discriminator")

                    mixup = True
                    both = False
                    if mixup or both:
                        sampled_mixed_states, sampled_mixed_actions, lam = mixup_state_action(
                            self, sampled_expert_states, sampled_expert_actions, sampled_generator_states, sampled_generator_actions
                        )
                        mixed_logits, _, _ = self.discriminator.act(
                            {"states": sampled_mixed_states, "taken_actions": sampled_mixed_actions}, role="discriminator"
                        )
                        # Mixup loss
                        mixup_loss = lam * nn.BCEWithLogitsLoss()(mixed_logits, torch.ones_like(mixed_logits)) + \
                                     (1 - lam) * nn.BCEWithLogitsLoss()(mixed_logits, torch.zeros_like(mixed_logits))
                
                    # Discriminator loss with expert logits high and generator logits low
                    discriminator_loss = 0.5 * (
                        nn.BCEWithLogitsLoss()(generator_logits, torch.zeros_like(generator_logits))
                        + torch.nn.BCEWithLogitsLoss()(expert_logits, torch.ones_like(expert_logits))
                    )
                    
                    if mixup:
                        discriminator_loss = mixup_loss
                    if both:
                        discriminator_loss += mixup_loss
                    
                    if self._discriminator_logit_regularization_scale:
                        logit_weights = torch.flatten(list(self.discriminator.modules())[-1].weight)
                        discriminator_loss += self._discriminator_logit_regularization_scale * torch.sum(
                            torch.square(logit_weights)
                        )

                    if self._discriminator_gradient_penalty_scale:
                        amp_motion_gradient = torch.autograd.grad(
                            expert_logits,
                            sampled_expert_states,
                            grad_outputs=torch.ones_like(expert_logits),
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True,
                        )
                        gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
                        discriminator_loss += self._discriminator_gradient_penalty_scale * gradient_penalty

                    if self._discriminator_weight_decay_scale:
                        weights = [
                            torch.flatten(module.weight)
                            for module in self.discriminator.modules()
                            if isinstance(module, torch.nn.Linear)
                        ]
                        weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
                        discriminator_loss += self._discriminator_weight_decay_scale * weight_decay

                    discriminator_loss *= self._discriminator_loss_scale
                    cumulative_discriminator_loss += discriminator_loss.item()
                    
            # Optimization step
            self.optimizer_disc.zero_grad()
            self.scaler_disc.scale(discriminator_loss).backward()
            if config.torch.is_distributed:
                self.discriminator.reduce_parameters()
                
            self.scaler_disc.step(self.optimizer_disc)
            self.scaler_disc.update()

            self.track_data(
                "Loss / Discriminator loss", cumulative_discriminator_loss / (self._mini_batches)
            )
                    
            
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # Compute Combined Rewards from Discriminator
        rewards = self.memory.get_tensor_by_name("rewards")
        states = self.memory.get_tensor_by_name("states")
        actions = self.memory.get_tensor_by_name("actions")
        self.reply_buffer.add_samples(
            actions=actions.permute(1, 0, *range(2, actions.ndim)).contiguous().view(-1, actions.shape[-1]),
            states=states.permute(1, 0, *range(2, states.ndim)).contiguous().view(-1, states.shape[-1]),
        )
        
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            disc_logits, _, _ = self.discriminator.act(
                {"states": self._amp_state_preprocessor(states), "taken_actions": self._amp_action_preprocessor(actions)}, role="discriminator"
            )
            style_reward = -torch.log(torch.clamp(1 - torch.sigmoid(disc_logits), min=1e-4))
            style_reward *= self._discriminator_reward_scale
            style_reward = style_reward.view(rewards.shape)
        
        combined_rewards = style_reward    
        self.track_data("Reward / Combined reward", combined_rewards.mean().item())
        self.track_data("Reward / Style reward", style_reward.mean().item())

        # Compute returns and advantages
        values = self.memory.get_tensor_by_name("values")
        next_values = self.memory.get_tensor_by_name("next_values")
        returns, advantages = compute_gae(
            rewards=combined_rewards,
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=next_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)
                
                
        # Perform learning steps for the generator
        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        sampled_batches = self.memory.sample_all(names=self.tensors_names, mini_batches=self._mini_batches)

        for epoch in range(self._learning_epochs):
            # mini-batches loop
            for batch_index, (
                sampled_states,
                sampled_actions,
                _,
                _,
                _,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                _,
            ) in enumerate(sampled_batches):

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

                    sampled_states = self._state_preprocessor(sampled_states, train=True)
                    _, next_log_prob, _ = self.policy.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
                    )
 
                    # Entropy loss
                    if self._entropy_loss_scale:
                        entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    # Policy loss
                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
                    # print(f"[DEBUG] surrogate: {surrogate.view(-1)}")
                    # print(f"[DEBUG] surrogate clipped: {surrogate_clipped.view(-1)}")
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
                    # print(f"[DEBUG] policy loss: {policy_loss.item()}")

                    # Value loss
                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                    if self._clip_predicted_values:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
                        )
                    value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # Optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    self.value.reduce_parameters()

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Advantage / Mean", advantages.mean().item())
        self.track_data("Advantage / Standard deviation", advantages.std().item())
        
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data(
                "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
            )
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
            
            
def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_dones = dones.logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        advantage = (
            rewards[i]
            - values[i]
            + discount_factor * (next_values[i] + lambda_coefficient * not_dones[i] * advantage)
        )
        advantages[i] = advantage
    # print(f"[DEBUG] Memory size: {memory_size}")
    # print(f"[DEBUG] rewards: {rewards.view(-1)}")
    # print(f"[DEBUG] values: {values.view(-1)}")
    # print(f"[DEBUG] advantages: {advantages.view(-1)}")
    # print(f"[DEBUG] adv mean: {advantages.mean()} | adv std: {advantages.std()}")
    
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # print(f"[DEBUG] Normalized advantages: {advantages.view(-1)}")
    return returns, advantages



def plot_trajectories(
    sampled_generator_states, sampled_generator_actions, sampled_expert_states, sampled_expert_actions,
    fig, ax
    ):
    n_states = sampled_expert_states.shape[1]
    n_actions = sampled_expert_actions.shape[1]

    for i in range(n_states):
        ax[i].clear()
        # ax[i].plot(sampled_amp_states[:, i].cpu().numpy(), label="AMP States")
        ax[i].plot(sampled_generator_states[:, i].cpu().numpy(), label="Generator States")
        ax[i].plot(sampled_expert_states[:, i].cpu().numpy(), label="Expert States")
        ax[i].set_title(f"State {i}")
        ax[i].legend()
        ax[i].set_xlabel("Time")
        ax[i].set_ylabel("Value")
        
    for i in range(n_actions):
        ax[n_states + i].clear()
        # ax[n_states + i].plot(sampled_amp_actions[:, i].cpu().numpy(), label="AMP Actions")
        ax[n_states + i].plot(sampled_generator_actions[:, i].cpu().numpy(), label="Generator Actions")
        ax[n_states + i].plot(sampled_expert_actions[:, i].cpu().numpy(), label="Expert Actions")
        ax[n_states + i].set_title(f"Actions {i}")
        ax[n_states + i].legend()
        ax[n_states + i].set_xlabel("Time")
        ax[n_states + i].set_ylabel("Value")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)