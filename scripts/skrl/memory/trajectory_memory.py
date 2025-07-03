import numpy as np
from skrl.memories.torch import Memory
from skrl.memories.torch.random import RandomMemory
# rom src.memory.random_memory import CustomRandomMemory
import torch
from typing import List, Tuple

class TrajectoryMemory(RandomMemory):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traj_start_index = [0]
    
    def add_samples(self, **tensors: torch.Tensor) -> None:
        """ 
        Patched version of Memory.add_samples() for self.num_envs = 1 and multi-sample 
        self.memory_index was incremented for each tensor name instead of one time per sample
        """
        if not tensors:
            raise ValueError(
                "No samples to be recorded in memory. Pass samples as key-value arguments (where key is the tensor name)"
            )

        # dimensions and shapes of the tensors (assume all tensors have the dimensions of the first tensor)
        tmp = tensors.get("states", tensors[next(iter(tensors))])  # ask for states first
        dim, shape = tmp.ndim, tmp.shape

        # multi environment (number of environments equals num_envs)
        if dim > 1 and shape[0] == self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index].copy_(tensor)
                # if name == "truncated":
                #     print("truncated", tensor)
            self.memory_index += 1
        # multi environment (number of environments less than num_envs)
        elif dim > 1 and shape[0] < self.num_envs:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index : self.env_index + tensor.shape[0]].copy_(
                        tensor
                    )
            self.env_index += tensor.shape[0]
        # single environment - multi sample (number of environments greater than num_envs (num_envs = 1))
        elif dim > 1 and self.num_envs == 1:
            # print("RANDOM MEMORY DEBUG ##### ", shape, self.memory_index, self.memory_size)
            start_memory_index = self.memory_index
            for name, tensor in tensors.items():
                self.memory_index = start_memory_index
                if name in self.tensors:
                    num_samples = min(shape[0], self.memory_size - self.memory_index)
                    remaining_samples = shape[0] - num_samples
                    # copy the first n samples
                    self.tensors[name][self.memory_index : self.memory_index + num_samples].copy_(
                        tensor[:num_samples].unsqueeze(dim=1)
                    )
                    self.memory_index += num_samples
                    # storage remaining samples
                    if remaining_samples > 0:
                        self.tensors[name][:remaining_samples].copy_(tensor[num_samples:].unsqueeze(dim=1))
                        self.memory_index = remaining_samples
        # single environment
        elif dim == 1:
            for name, tensor in tensors.items():
                if name in self.tensors:
                    self.tensors[name][self.memory_index, self.env_index].copy_(tensor)
            self.env_index += 1
        else:
            raise ValueError(f"Expected shape (number of environments = {self.num_envs}, data size), got {shape}")

        # update indexes and flags
        if self.env_index >= self.num_envs:
            self.env_index = 0
            self.memory_index += 1
        if self.memory_index >= self.memory_size:
            self.memory_index = 0
            self.filled = True

            # export tensors to file
            if self.export:
                self.save(directory=self.export_directory, format=self.export_format)


    def sample(
        self, names: Tuple[str], batch_size: int, mini_batches: int = 1, sequence_length: int = 1
    ) -> List[List[torch.Tensor]]:
        """Sample a batch from memory randomly

        :param names: Tensors names from which to obtain the samples
        :type names: tuple or list of strings
        :param batch_size: Number of element to sample
        :type batch_size: int
        :param mini_batches: Number of mini-batches to sample (default: ``1``)
        :type mini_batches: int, optional
        :param sequence_length: Length of each sequence (default: ``1``)
        :type sequence_length: int, optional

        :return: Sampled data from tensors sorted according to their position in the list of names.
                 The sampled tensors will have the following shape: (batch size, data size)
        :rtype: list of torch.Tensor list
        """
        # compute valid memory sizes
        size = len(self)
        if sequence_length > 1:
            sequence_indexes = torch.arange(0, self.num_envs * sequence_length, self.num_envs)
            size -= sequence_indexes[-1].item()

        # generate random indexes
        if self._replacement:
            if size // sequence_length > 0:
                indexes = torch.randint(0, (size // sequence_length), (batch_size,)) * sequence_length
            else:
                indexes = torch.tensor([0] * batch_size)
        else:
            # details about the random sampling performance can be found here:
            # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
            indexes = torch.randperm(size, dtype=torch.long)[:batch_size]
        # print(sequence_indexes)
        # print(indexes)
        # generate sequence indexes
        if sequence_length > 1:
            indexes = (sequence_indexes.repeat(indexes.shape[0], 1) + indexes.view(-1, 1)).view(-1)
            # indexes = (indexes.repeat(sequence_indexes.shape[0], 1) + sequence_indexes.view(-1, 1)).view(-1)
            # print(indexes)
            # print(indexes2)

        self.sampling_indexes = indexes
        return self.sample_by_index(names=names, indexes=indexes, mini_batches=mini_batches)
    
    
    
if __name__ == "__main__":
    # Example usage
    num_envs = 1
    memory = TrajectoryMemory(num_envs=num_envs, memory_size=548*num_envs)
    print("Memory size:", memory.memory_size)
    print("Number of environments:", memory.num_envs)
    print("Memory index:", memory.memory_index)
    
    memory.create_tensor(name="states", size=3)
    memory.create_tensor(name="actions", size=2)
    memory.create_tensor(name="rewards", size=1)
    
    # Create some sample data
    states = torch.randn(num_envs, 3)
    actions = torch.randn(num_envs, 2)
    rewards = torch.randn(num_envs, 1)
    
    # Add samples to memory
    for _ in range(429):
        memory.add_samples(states=states, actions=actions, rewards=rewards)
    print("Memory index after adding samples:", memory.memory_index)
    
    res = memory.sample(names=["states", "actions"], batch_size=2, mini_batches=1, sequence_length=100)
    print(res[0][0].shape)
    print(memory.sampling_indexes)


def load_memory(file: str, max_trajectories: int = 1000) -> TrajectoryMemory:
    data = np.load(file, allow_pickle=True)
    key_shapes = {key: data[key].shape for key in data.files}

    first_key = next(iter(data.files))
    first_shape = data[first_key].shape
    
    traj_length = first_shape[0]
    num_trajectories = first_shape[1]
    num_trajectories = min(num_trajectories, max_trajectories)

    memory = TrajectoryMemory(
        num_envs=1,
        memory_size=traj_length * num_trajectories,
    )
    
    print(f"Loaded Memory with {memory.memory_size} samples and {memory.num_envs} environments")
    print(f"Trajectory length: {traj_length}, Number of trajectories: {num_trajectories}")

    for key, shape in key_shapes.items():
        memory.create_tensor(key, size=shape[-1])
    for key in data.files:
        for i in range(num_trajectories):
            mem_data = torch.from_numpy(data[key][:, i]).squeeze(0) 
            memory.add_samples(**{key: mem_data})
    return memory