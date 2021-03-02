from enum import Enum
from typing import Callable, List
import torch
from torch import Tensor

import gym
import ma_gym

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
