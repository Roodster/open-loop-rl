"""
    Buffer:
        Summarizes transitions and builds an experience replay buffer.
"""

from typing import Union, NamedTuple, Optional

import numpy as np
import torch as th
from gymnasium import spaces
import torch as th
import threading

class TrajectoryBufferBatch(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class Episode:
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': []
    }
        
    def add(self, state, action, reward, done):
        """
            Add step to episode in progress for trajectory buffer. An episode is only added to the replay buffer when it is completed.
        """
    
        self.data['states'].append(state)
        self.data['actions'].append(action)
        self.data['rewards'].append(reward)
        self.data['dones'].append(done)      
        

    def get_episode_and_reset(self):
        episode = self.data.copy()
        
        self.reset()
        return episode
        

class TrajectoryBuffer():
    """
    Replay buffer that samples trajectories instead of single state action pairs.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space of a single environment
    :param action_space: Action space of a single environment
    :param max_episode_length
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param trajectory_length: Number of timesteps included in each data point sampled from the buffer i.e. length of the trajectory
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_starts: np.ndarray
    dones: np.ndarray
    values: np.ndarray

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            batch_size,
            use_last_episode,
            max_episode_length: int,
            device: Union[th.device, str] = "auto",
            n_envs: int = 1,
            trajectory_length=10
    ):
        self.lock = threading.Lock()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = observation_space.shape # type: ignore[assignment]

        self.action_dim = int(np.prod(action_space.shape))
        self.pos = 0
        self.full = False
        self.device = device
        self.n_envs = n_envs
        self.actual_episode_lengths = None
        self.max_episode_length = max_episode_length
        self.trajectory_length = trajectory_length
        # we add trajectory_length to the max_episode_length to make sure that the episode has enough steps at the end
        # such that agent can also learn from when it is at the end of the maze
        self.padded_episode_length = max_episode_length + trajectory_length
        self.batch_size = batch_size
        self.use_last_episode = use_last_episode
        self.reset()
        

    def reset(self) -> None:
        """
        Resets the buffer arrays to zero:
            - observations, actions, rewards, episode starts, values, dones, episode length
        For the buffer to later store and access o, a, r, v.. for each episode, across all environments, and for each timestep within those episodes
        """

        # (buffer_size=1000, episode_dim=210, n_envs=1, obs_shape=84, obs_shape=84, obs_shape=3)
        # obs_shape of individual observation = (84, 84, 3)
        self.observations = np.zeros((self.buffer_size, self.padded_episode_length, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.padded_episode_length, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.padded_episode_length, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.padded_episode_length, self.n_envs), dtype=np.float32)
        self.actual_episode_lengths = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.pos = 0
        self.full = False
        
        
    def add(
            self,
            obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            dones: np.ndarray,
    ) -> None:
        """
        Adds a new trajectory to the buffer, padding each component to the padded_episode_length.
        Padding makes sure that we can sample trajectories from the end of the episode without being out of bounds.
        This is important to allow the agent to learn from the end of the episodes, where trajectory_length steps can
        not be taken.

        Args:
            action: non-padded action array
            obs: non-padded obs array
            reward: non-padded reward array
            episode_start: non-padded episode_start array. marks the start of the episode
            value: non-padded value array

        """
        # self.pos = current position in the buffer where the episode will be stored
        self.lock.acquire()
        try:
            self.observations[self.pos] = self._pad_array_to_length(np.array(obs).copy(), self.padded_episode_length)
            self.actions[self.pos] = self._pad_array_to_length(np.array(action).copy(), self.padded_episode_length)
            self.rewards[self.pos] = self._pad_array_to_length(np.array(reward).copy(), self.padded_episode_length)

            self.dones[self.pos] = np.zeros_like(self._pad_array_to_length(np.array(reward).copy(), self.padded_episode_length))
            # Handles the padded area: the padded part should be marked as terminal states (dones)
            # select the part of the dones corresponding to the padding region (from the end of the episode = obs.shape[0] to the total padded length).
            self.dones[self.pos, obs.shape[0]:] += 1

            self.actual_episode_lengths[self.pos] = obs.shape[0]

            # same as in base buffer class, if buffer is full wrap around
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0
        finally: self.lock.release()


    def add_episode(self, episode):
        """
            Add episode to trajctory buffer.
                - observation, action, reward, episode start, value
            for each step in an episode.
            The episode should replaced with an empty episode after calling this function.
            Args:
                episode: the episode that we want to add to our trajectory buffer.
        """

        obs = th.tensor(np.array(episode['states']))
        action = th.tensor(np.array(episode['actions']))
        action = action.view(*action.shape, 1)
        reward = th.tensor(np.array(episode['rewards']))
        dones = th.tensor(np.array(episode['dones']))
        # start = th.tensor(np.array(episode['episode_start']))
        # start = start.view(*start.shape, 1)
        # value = th.tensor(episode['value'])
        # value = value.view(*value.shape, 1)
        self.add(obs, action, reward, dones)
        

    def sample_trajectories(
            self) -> TrajectoryBufferBatch:
        """
        Method that samples a batch of trajectories.
        Args:
            batch_size: size of the batch to sample
        Returns: TrajectoryBufferBatch
        """
        self.lock.acquire()
        try:
            upper_bound = self.buffer_size if self.full else self.pos
            
            if self.use_last_episode:
                batch_indexes = np.random.randint(0, upper_bound, size=self.batch_size)
                batch_indexes[-1] = self.pos - 1             
            else:
                batch_indexes = np.random.randint(0, upper_bound, size=self.batch_size)
        finally: self.lock.release()
        
        return self._get_batch_trajectories(batch_indexes)

    def _get_batch_trajectories(
            self,
            selected_episodes: np.ndarray) -> TrajectoryBufferBatch:
        """
        Retrieves and processes trajectories based on sampled batch_indexes.
        Extracts a segment of obs/act/rew/etc. from specific episodes (selected_episodes[i]) and environment (env_indices[i])
        Creates the final obs/act/rew/etc. by appending the selected fragments of the trajectories
        Args:
            selected_episodes: the selected episodes to sample trajectories from
        Returns: TrajectoryBufferBatch
        """
        # what the final TrajectoryBufferBatch will contain
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        # Generate indices for the selected episodes and environments
        env_indices = np.random.randint(0, self.n_envs, size=(len(selected_episodes),))
        length_of_selected_episodes = self.actual_episode_lengths[selected_episodes]
        traj_start = np.random.randint(0, length_of_selected_episodes, size=(length_of_selected_episodes.shape[0],))

        # Create a range array for trajectory lengths
        traj_range = np.arange(self.trajectory_length)

        # Use broadcasting to create indices for the entire batch
        batch_indices = selected_episodes[:, None]  # Shape: (batch_size, 1)
        env_indices = env_indices[:, None]  # Shape: (batch_size, 1)
        traj_indices = traj_start[:, None] + traj_range  # Shape: (batch_size, trajectory_length)

        # Gather observations, actions, rewards, and dones using advanced indexing
        observations = self.observations[batch_indices, traj_indices, env_indices]
        next_observations = self.observations[batch_indices, traj_indices + 1, env_indices]
        actions = self.actions[batch_indices, traj_indices, env_indices, :]
        rewards = self.rewards[batch_indices, traj_indices, env_indices]
        dones = self.dones[batch_indices, traj_indices, env_indices]

        # Wrap up all the data for the trajectory
        data = (
            np.array(observations),
            np.array(actions),
            np.array(next_observations),
            np.expand_dims(np.array(dones), -1),
            np.expand_dims(np.array(rewards), -1),
        )

        return TrajectoryBufferBatch(*tuple(map(self.to_torch, data)))


    def _pad_array_to_length(self, array, total_length):
        """
        Pad the array with zeros at the end to make sure that the agent learns from the end of the maze too and that the
        trajectories are sampled from there as well.
        Args:
            array: the array of obs/actions/etc. to be padded at the end with zeros
            total_length: the final length of the padded array

        Returns:

        """

        # *array.shape[1:] means if array is (episode_length, obs_dim1, obs_dim2) -> (obs_dim1, obs_dim2)
        # we want the same dim for each timestamp of the padded_array as for the array
        padded_array = np.zeros((total_length, *array.shape[1:]))
        # such that to only have zeros at the end
        padded_array[:array.shape[0]] = array
        return padded_array
    
    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)