from typing import Any, Callable, List, Sequence, Tuple, Type, Union
from collections import OrderedDict
import gym
import ma_gym as _
import numpy as np
from stable_baselines3.common.type_aliases import GymEnv
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import obs_space_info

class MultiAgentEnvWrapper(VecEnv): 
    """
    """
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        """
        """
        pass 

    def reset(self) -> VecEnvObs:
        pass 

    def step_async(self, actions: np.ndarray) -> None:
        pass 

    def step_wait(self) -> VecEnvStepReturn:
        pass

    def close(self) -> None:
        pass
    def get_attr(self, attr_name: str, indices: VecEnvIndices) -> List[Any]:
        pass
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices, **method_kwargs) -> List[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices) -> List[bool]:
        pass
    
    def get_images(self) -> Sequence[np.ndarray]:
        pass

    def render(self, mode: str) -> Optional[np.ndarray]:
        return super().render(mode=mode)
    
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
    def __init__(self, env: Union[str, GymEnv]):
        if isinstance(env, str):
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
