from enum import Enum
from typing import Callable, List
from itertools import chain 

import torch
from torch import Tensor
import gym
import ma_gym
import numpy as np

class td3_parameters(Enum):
    LEARNING_RATE       = 0.001
    BUFFER_SIZE         = 100000
    LEARNING_STARTS     = 100
    BATCH_SIZE          = 100
    TAU                 = 0.005
    GAMMA               = 0.99
    GRADIENT_STEPS      = -1 #DISABLE
    N_EPISODES_ROLLOUT  = 1
    POLICY_DELAY        = 2


class training_paramters(Enum):
    DEFAULT_LEARN_STEPS  = 10000
    DEFAULT_TEST_STEPS   = 10000


class nn_structure(Enum):
    ACTOR_FC_NODES  = 256
    CRITIC_FC_NODES = 256


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
        - Addressed the attribute error with the original observation space of ma-gym
          by "concatenating" the first dimensions
    TODO:
        - Process action space
        - Add `reward range` attribute
        - Other necessary attributes of ma-gym env
    """
    def __init__(self, env: str):
        env = gym.make(env)
        self.agent_num = len(env.action_space)
        self.observation_space_ = env.observation_space[0]
        self.observation_space = self.wrap()
        self.action_space = env.action_space

    def wrap(self):
        obs_shape = self.observation_space_.shape
        obs_shape_ = [obs_shape[i]*self.agent_num if i == 0 else obs_shape[i] for i in range(len(obs_shape))]
        obs_shape_ = tuple(obs_shape_)
        obs_low = np.array(self.observation_space_.low).flatten()[0]
        obs_high = np.array(self.observation_space_.high).flatten()[0]
        return gym.spaces.Box(obs_low, obs_high, obs_shape_)


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


def pond_duel_action_value_converter(t:Tensor) -> Tensor:
    # $t \in (-1,1)$
    t = t + 1
    # $t \in (0,2)$
    return torch.round(t)


pong_duel_adapter = action_adapter(converter = pond_duel_action_value_converter)
