from torch import Tensor
import gym
import ma_gym as _
import numpy as np
import torch


class env_wrapper:
    """
    A wrapper for ma-gym environments

    Args:
        env: name of the ma-gym environment (str)

    - Compatible with observation space of high dimension
    - Assume the type of observation space is gym.spaces.Box 
      (more types can be added at a later stage)
    - Assume discrete action space
      (can add supports for continuous one at a later stage)
    
    Implemented:
        - Addressed the attribute error with the original observation space 
          of ma-gym by "concatenating" the first dimensions
        - Convert discrete action space to continuous one. 
        - Add reward_range & metadata attributes
    """
    def __init__(self, env: str):
        env = gym.make(env)
        self.env = env

        self.agent_num = len(env.action_space)

        self.observation_space_ = env.observation_space[0]
        self.observation_space = self.wrap_obs()

        self.action_space_ = env.action_space[0]
        self.action_space = self.wrap_act()

        self.reward_range = env.reward_range
        self.metadata = env.metadata
        
        self.state_dim = self.observation_space_.shape[0]
        self.action_dim = 1 # Discrete

    def wrap_obs(self):
        obs_shape = self.observation_space_.shape
        obs_shape_ = [self.agent_num if i == -1 else obs_shape[i] for i in range(-1,len(obs_shape))]
        obs_shape_ = tuple(obs_shape_)
        obs_low = np.array(self.observation_space_.low).flatten()[0]
        obs_high = np.array(self.observation_space_.high).flatten()[0]

        return gym.spaces.Box(obs_low, obs_high, obs_shape_)

    def wrap_act(self):
        act_num = self.action_space_.n
        return gym.spaces.Box(-1, 1, (self.agent_num, 1))

    def reset(self, **kwargs):
        states_ = self.env.reset()
        states = []
        for state in states_:
            states.append(state)
        return states

    def step(self, actions):
        actions = Tensor(actions)
        # action is of shape(2, 1), varied within (-1.0, 1,0)
        actions_int = []
        ## with torch.no_grad:
        actions_int.extend(torch.round(actions * 1.5 + 1).flatten().tolist())

        states, rewards, dones, info = self.env.step(actions_int)
        # state: list of Tensor(10,), required Tensor(2,10)
        # rewards: list of int, required Tensor(1,2)
        states = [Tensor(state) for state in states]
        return torch.stack(states), torch.Tensor(rewards).reshape(-1, len(rewards)), any(dones), info
