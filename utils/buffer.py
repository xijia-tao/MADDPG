from stable_baselines3.common.buffers import ReplayBuffer
from gym import spaces
from typing import Union, Optional
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize, 
import torch as th
import numpy as np

class MultiAgentReplayBuffer(ReplayBuffer):
    """
    """

    def __init__(self,
    buffer_size:       int,
    observation_space: spaces.Space,
    action_space:      spaces.Space,
    device:            Union[th.device, str],
    n_envs:            int,
    n_agent:           int,
    optimize_memory_usage: bool):
        """
        """
        super().__init__(
            buffer_size, 
            observation_space, 
            action_space, 
            device=device, 
            n_envs=n_envs, 
            optimize_memory_usage=optimize_memory_usage
        )
        self.rewards = np.zeros((self.buffer_size, n_envs, n_agent), dtype=np.float32)
        self.dones   = np.zeros((self.buffer_size, n_envs, n_agent), dtype=bool)
        # dones of shape [100, 1, 1]
        # the original replay buffer dones is of shape [100, 1], which leads to unexpected broadcasting
        # during its multiplication with Q, as both Q and reward are of shape [100, 1, agent_num]


    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        # dones remains to be a np.array
        # because Tensor does not support substraction of boolean types
        # but numpy does
        data = (
            self.to_torch(self._normalize_obs(self.observations[batch_inds, 0, :], env)),
            self.to_torch(self.actions[batch_inds, 0, :]),
            self.to_torch(next_obs),
            self.dones[batch_inds],
            self.to_torch(self._normalize_reward(self.rewards[batch_inds], env)),
        )
        return ReplayBufferSamples(*data)