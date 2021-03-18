import numpy as np
import torch as th

from gym import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from typing import Union, Optional

class MultiAgentReplayBuffer(ReplayBuffer):
    """ Replay buffer for multi-agent scenario
    
    The shape for the arrays that storing the intercations are expanded to extra dimension
    so that multiple agents can be supervised all together
    """

    def __init__(self,
    buffer_size:       int,
    observation_space: spaces.Space,
    action_space:      spaces.Space,
    device:            Union[th.device, str],
    n_envs:            int,
    n_agent:           int,
    optimize_memory_usage: bool):
        """ Constructor
        
        Args: 
            buffer_size: size of the buffer
            observation_space: the aggregated observation space for all the agents
            action_space: the aggregated continuous action space for all the agents
            device: device descriptor
            n_envs: number of environments, from which interactions will be collected
                simultaneously
            n_agent: number of agents, which must be the same for EVERY envrionment
                that shares the buffer
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


    # overriden
    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        # dones remains to be a np.array
        # because Tensor does not support substraction of boolean types
        # but numpy does
        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            self.dones[batch_inds].astype(int),
            self._normalize_reward(self.rewards[batch_inds], env)
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
