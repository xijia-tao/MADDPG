from collections import OrderedDict
import copy
from typing import Any, Callable, List, Optional, Sequence, Type, Union
import gym
import ma_gym as _
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import copy_obs_dict, obs_space_info, dict_to_obs

class VectorizedMultiAgentEnvWrapper(VecEnv): 
    """ The wrapper for multi agent environment

    This wrapping the MA env(s) into a VecEnv, s.t. it(they) can 
    perform in the compatible way with those models defiend for
    single agent environments. 
    """
    def __init__(self, env_fns: List[Callable[[], Union[str, gym.Env]]]):
        """ Constructor

        Args:
             env_fns: a list of functions that return environments to vectorize
        """
        self.envs = [self.MultiAgentEnvWrapper(fn()) for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.agent_num = env.agent_num
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs, 1, env.agent_num), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        """ Save the observation of a certain ma-env to obs_buf

        Args: 
            env_idx: index of the env whose observation is to be saved
            obs: the vectorized env observations
        """
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    # overriden
    def reset(self) -> VecEnvObs:
        for env_idx, each_env in enumerate(self.envs):
            obs = each_env.reset()
            self._save_obs(env_idx, obs)
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    # overriden
    def step_async(self, actions: np.ndarray) -> None:
        # store the action
        self.actions = actions

    # overriden
    def step_wait(self) -> VecEnvStepReturn:
        # wield the stored action for env.step
        for idx, env in enumerate(self.envs):
            obs, self.buf_rews[idx], self.buf_dones[idx], self.buf_infos[idx] = env.step(
                self.actions[idx]
            )
            if self.buf_dones[idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[idx]["terminal_observation"] = obs
                obs = self.envs[idx].reset()
            self._save_obs(idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), copy.deepcopy(self.buf_infos))

    # overriden
    def close(self) -> None:
        for env in self.envs: env.close()

    # overriden
    def get_attr(self, attr_name: str, indices: VecEnvIndices) -> List[Any]:
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    # overriden
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices) -> None:
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    # overriden
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices, **method_kwargs) -> List[Any]:
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    # overriden
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices) -> List[bool]:
        target_envs = self._get_target_envs(indices)
        from stable_baselines3.common import env_util
        return [env_util.is_wrapped(each_env, wrapper_class) for each_env in target_envs]
    
    # overriden
    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    # overriden
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if self.num_envs == 1:
            return self.envs[0].render(mode)
        else:
            return super().render(mode="rgb_array")

    # overriden
    def seed(self, seed: Optional[int]) -> List[Union[None, int]]:
        return [env.seed(seed + idx) for idx, env in enumerate(self.envs)]

    # overriden
    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    
    class MultiAgentEnvWrapper:
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
        def __init__(self, env: Union[str, gym.Env]):
            if isinstance(env, str):
                env = gym.make(env)

            self.env = env

            self.agent_num = len(env.action_space)

            self.observation_space_ = env.observation_space[0]
            self.observation_space = self._wrap_obs()

            self.action_space_ = env.action_space[0]
            self.action_space = self._wrap_act()

            self.reward_range = env.reward_range
            self.metadata = env.metadata
            
            self.state_dim = self.observation_space_.shape[0]
            # action_space_: Discrete(3)
            self.action_dim = 1

        def _wrap_obs(self):
            obs_shape = self.observation_space_.shape
            obs_shape_ = [self.agent_num if i == -1 else obs_shape[i] for i in range(-1,len(obs_shape))]
            obs_shape_ = tuple(obs_shape_)
            obs_low = np.array(self.observation_space_.low).flatten()[0]
            obs_high = np.array(self.observation_space_.high).flatten()[0]

            return gym.spaces.Box(obs_low, obs_high, obs_shape_)

        def _wrap_act(self):
            act_num = 1  # self.action_space_.n
            return gym.spaces.Box(-1, 1, (self.agent_num, act_num))

        def reset(self, **kwargs):
            states_ = self.env.reset(kwargs)
            states = []
            for state in states_:
                states.append(state)
            return states

        def seed(self, i: int):
            return self.env.seed(i)

        def render(self, mode: str = "human"):
            return self.env.render(mode)

        def close(self):
            return self.env.close()

        def reset(self):
            return self.env.reset()

        def step(self, actions):
            actions = torch.Tensor(actions)
            # action is of shape(2, 1), varied within (-1.0, 1,0)
            actions_int = []
            ## with torch.no_grad:
            actions_int.extend(torch.round(actions * 1.5 + 1).flatten().tolist())

            states, rewards, dones, info = self.env.step(actions_int)
            # state: list of Tensor(10,), required Tensor(2,10)
            # rewards: list of int, required Tensor(1,2)
            states = [torch.Tensor(state) for state in states]
            return torch.stack(states), torch.Tensor(rewards).reshape(-1, len(rewards)), any(dones), info
