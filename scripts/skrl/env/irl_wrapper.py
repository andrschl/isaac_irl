import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from typing import List
import torch

from scripts.skrl.memory.trajectory_memory import load_memory

class ExpertTrajectoriesWrapper(Wrapper):
    def __init__(self, env, expert_data_path: str, max_trajectories: int = 1000):
        super().__init__(env)
        self._motion_loader = load_memory(expert_data_path, max_trajectories=max_trajectories)
        
    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> List[List[torch.Tensor]]:
        motion_batches = self._motion_loader.sample(
            names=('states', 'actions'),
            batch_size=1,
            sequence_length=num_samples,
        )
        return motion_batches
