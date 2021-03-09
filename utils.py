from typing import Callable, List
from itertools import chain 

import torch
from torch import Tensor
import gym
import ma_gym
import numpy as np


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
        obs_shape_ = [obs_shape[i]*self.agent_num if i == 0 else obs_shape[i] for i in range(len(obs_shape))]
        obs_shape_ = tuple(obs_shape_)
        obs_low = np.array(self.observation_space_.low).flatten()[0]
        obs_high = np.array(self.observation_space_.high).flatten()[0]

        return gym.spaces.Box(obs_low, obs_high, obs_shape_)

    def wrap_act(self):
        act_num = self.action_space_.n
        return gym.spaces.Box(-0.5, act_num-0.5, (self.agent_num, ))

    def reset(self, **kwargs):
        states_ = self.env.reset()
        states = []
        for state in states_:
            states.extend(state)
        return states

    def step(self, actions):
        if type(actions) != Tensor:
            # round randomly sampled actions from action space
            actions = [round(action) for action in actions] 

        states_, rewards, dones, info = self.env.step(actions)
        states = []
        for state in states_:
            states.extend(state)
        return states, rewards, dones, info #TODO


class action_adapter:
    """ A functional object, adapting the action"S" from policy to what can be accepted by the env

    When calling the object, the input list of actions of shape (batch_size, action_dim) will be 
    concatenated to a single tensor of shape (batch_size, action_dim * n), where n is the len of
    the input list, i.e. the number of agents. 
    
    Meanwhile, the object will as well call the converter which shall be a function from Tensor to
    Tensor, to convert the action from the policy output to the range that can be accepted by the
    environment FOR EACH AGENT's action. The converter MAY change the dimension of action. 

    Args
        converter: the function to convert between action spaces. 
    """
    def __init__(self, converter: Callable[[Tensor], Tensor] = None) -> None:
        self._converter    = converter

    def __call__(self, input: List[Tensor]) -> Tensor:
        result = []
        for each in input:
            result.append(self._converter(each) if self._converter is not None else each)

        concated = torch.cat(result, 1)
        return concated


def pong_duel_action_value_converter(t:Tensor) -> Tensor:
    # $t \in (-1,1)$
    t = 1.5*t + 1
    # $t \in (-0.5, 2.5)$
    return torch.round(t)


pong_duel_adapter = action_adapter(converter = pong_duel_action_value_converter)