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

from scripts.skrl.memory.random_memory import CustomRandomMemory

# fmt: off
# [start-config-dict-torch]
BC_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 6,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch
    "n_policy_updates": 1,   # number of discriminator updates during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 5e-5,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    
    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.0,              # clipping coefficient for the norm of the gradients

    "batch_size": 512,                  # batch size for updating the reference motion dataset

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


class BC(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Memory,
        observation_space: Union[int, Tuple[int], gymnasium.Space],
        action_space: Union[int, Tuple[int], gymnasium.Space],
        device: Union[str, torch.device],
        cfg: dict,
        motion_dataset: Memory,
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
        _cfg = copy.deepcopy(BC_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )
        
        self.observation_space: Union[int, Tuple[int], gymnasium.Space]
        self.action_space: Union[int, Tuple[int], gymnasium.Space]
        self.motion_dataset = motion_dataset

        # models
        self.policy: Model = self.models.get("policy", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()

        # PLotting
        plt.ion()
        self.fig, self.ax = plt.subplots(5, 1, figsize=(10, 10))
        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self.n_policy_updates = self.cfg["n_policy_updates"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._batch_size = self.cfg["batch_size"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]
        self._mixed_precision = self.cfg["mixed_precision"]
        self.collect_reference_motions = collect_reference_motions
        self.collect_observation = collect_observation
        self._rewards_shaper = self.cfg["rewards_shaper"]



        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
            self.scaler_disc = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)
            self.scaler_disc = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        print(f"Learning rate: {self._learning_rate} ytpe: {type(self._learning_rate)}")
        if self.policy is not None:
            self.optimizer = torch.optim.Adam(
                itertools.chain(self.policy.parameters(),),
                lr=self._learning_rate,
            )
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(
                    self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["amp_state_preprocessor_kwargs"])
            self.checkpoint_modules["_state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
            

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
            self.motion_dataset.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.motion_dataset.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            for i in range(math.ceil(self.motion_dataset.memory_size / self._batch_size)):
                if i % 10000 == 0:
                    print(f"Collecting reference motions {self.motion_dataset.memory_index}")
                motions = self.collect_reference_motions(self._batch_size)
                self.motion_dataset.add_samples(states=motions[0][0], actions=motions[0][1])
                
        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_states = None

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> tuple:
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
        """Record an environment transition in memory"""
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

            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
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
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        if self.collect_observation is not None:
            self._current_states = self.collect_observation()
            
        # Before interacting with the environement, update the actor model 
        self.set_mode("train")
        self._update_policy_epoch(timestep, timesteps)
        self.set_mode("eval")

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        self._rollout += 1
        # if not self._rollout % self._rollouts and timestep >= self._learning_starts:
        super().post_interaction(timestep, timesteps) # write tracking data and checkpoints
        return

        
    def _update_policy_epoch(self, timestep: int, timesteps: int) -> None:
        for i in range(self.n_policy_updates):
            sampled_expert_batches = self.motion_dataset.sample(
                names=["states", "actions"],
                batch_size=self.memory.num_envs,
                mini_batches=self._mini_batches,
                sequence_length=self._batch_size,
            )
            
            cumulative_policy_loss = 0
            for batch_index, expert_sample in enumerate(sampled_expert_batches):
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):                
                    sampled_expert_states = self._state_preprocessor(expert_sample[0], train=True)
                    sampled_expert_actions = expert_sample[1]             
                    
                    _, log_prob, _ = self.policy.act({
                        "states": sampled_expert_states,
                        "taken_actions": sampled_expert_actions
                    })

                    policy_loss = -log_prob.mean()
                    
                # Optimization step
                self.optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()
                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                
            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])