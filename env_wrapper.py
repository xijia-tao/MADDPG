from collections import OrderedDict
import copy
import gym
import ma_gym as _
import numpy as np
import torch

from gym import spaces
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import copy_obs_dict, obs_space_info, dict_to_obs

class VectorizedMultiAgentEnvWrapper(VecEnv): 
    """ The wrapper for multi agent environment

    This wrapping the MA env(s) into a VecEnv, s.t. it(they) can 
    perform in the compatible way with those models defiend for
    single agent environments. 
    """
    def __init__(self, env_fns: List[Tuple[Callable[[], Union[str, gym.Env]], Callable[[torch.Tensor], Union[torch.Tensor, np.ndarray, list]]]]):
        """ Constructor

        Args:
            env_fns: a list of pairs of functions, the first of which create one environment and the second is the mapper
                that converts the action outputs from the model to what can be accepted by the env object. 
                See VectorizedMultiAgentEnvWrapper.MultiAgentEnvWrapper for details
        """
        self.envs = [self.MultiAgentEnvWrapper(fn_pair[0](), fn_pair[1]) for fn_pair in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs, 1, env.agent_num), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def agent_num(self, indices: VecEnvIndices = 0) -> Union[int, List[int]]:
        indices = self._get_indices(indices)
        nums = [self.envs[i].agent_num for i in indices]
        if len(nums) == 1:
            return nums[0]
        else:
            return nums


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
        return (dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs)), np.copy(self.buf_rews), np.copy(self.buf_dones), copy.deepcopy(self.buf_infos))

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

    
    class MultiAgentEnvWrapper(gym.Env):
        """ Direct wrapper for EACH multi-agent environment. 

        We have this wrapper to ensure that each ENV outputs the multi-agent obseravation and 
        accept the corresponding action in form of Tensors, instead of list of arrays or
        tensors. Accordingly, the observation space and action space are converted to ensure
        the models which are designed for single agent env will be compitable with the ma ones.
        """
        def __init__(self, env: Union[str, gym.Env], mapper: Callable[[torch.Tensor], Union[torch.Tensor, np.ndarray, list]] = None):
            """ Constructor

            Args:
                env: the environment to be wrapped; either a string if the env has been registered to
                    gym, or the gym.Env object. 
                mapper: a function object, which maps the action outputs from the model to what can be
                    accepted by the env object. e.g. A TD3 outputs a tensor of float32 values which
                    vary within (-1.0, 1.0), and environment like Pone-Duel requires list of integers,
                    whose value only be 1, 2 or 3. A mapper will be needed to resolve this gap. 
            """
            if isinstance(env, str):
                env = gym.make(env)

            self.env       = env
            self.mapper    = mapper if mapper is not None else lambda action: action
            self.agent_num = len(env.action_space)

            # NOTE: the env.observation_space is a list, the i-th element corresponding to
            # the i-th agent's own observation space. 
            self.observation_space_ = env.observation_space[0] # the obs space for the first agent, as the representative
            self.observation_space  = self._wrap_obs() # the obs space for the ENTRIE multi-agent env, after wrapped

            self.action_space_ = env.action_space[0]
            self.action_space  = self._wrap_act()

            self.reward_range = env.reward_range
            self.metadata     = env.metadata

        def _wrap_obs(self) -> spaces.Box:
            """ Wrap the observation space

            The method must be called AFTER self.observation_space_ is set, and this method
            prepare the self.observation_space (with no underscore as its suffix). In the
            wrapping, one dimension will be added to the observation space as its first dimension
            which represents the number of agents. 

            Returns: 
                An box space, whose shape is (agent num, the shape of the original
                self.observation_space_)
            """
            obs_shape  = self.observation_space_.shape
            obs_shape_ = [self.agent_num if i == -1 else obs_shape[i] for i in range(-1,len(obs_shape))]
            obs_shape_ = tuple(obs_shape_)
            obs_low    = np.array(self.observation_space_.low).flatten()[0]
            obs_high   = np.array(self.observation_space_.high).flatten()[0]

            return spaces.Box(obs_low, obs_high, obs_shape_)

        def _wrap_act(self) -> spaces.Box:
            """ Wrap the action space. 

            Similar to self._wrap_obs. This method must be called AFTER self.action_space_ i set. 
            Preparing for the self.action_space, it first identifies the type of EACH single agent's
            own action space, and create the acceptable action space. 

            NOTE: 
                Due to the limitation of TD3 model, we treat the discrete space as a special case of
                Box whose shape = (1,)

            Returns:
                A box space, whose shape is (agent num,) if the action space is discrete, and
                (agent num, the shape of the original self.action_space_) otherwise. 
            """
            # calculate the action dimension
            if isinstance(self.action_space_, spaces.Discrete):
                act_shape = (self.agent_num,) # Discrete => ONE
            elif isinstance(self.action_space_, spaces.Box):
                act_shape = (self.agent_num,) + self.action_space_.shape
            elif isinstance(self.action_space_, list):
                act_shape = (self.agent_num, len(self.action_space_),) + self.action_space_[0].shape if isinstance(self.action_space_[0], spaces.Box) else (self.agent_num, len(self.action_space_)) 
            else:
                act_shape = (1,) # DEFAULT
            return spaces.Box(-1, 1, act_shape)

        def reset(self) -> np.ndarray:
            """ Reset the environemnt

            Returns: 
                The intialized state, re-formed into a tensor the same shape as self.observation_space
            """
            states_ = self.env.reset()
            states = []
            for state in states_:
                states.append(state)
            return torch.Tensor(states)

        def seed(self, i: int) -> None:
            """ Set the seeds for the inner environment

            Args:
                i: the int for seeding the env
            """
            return self.env.seed(i)

        def render(self, mode: str = "human") -> Any:
            """ Render the current state of environment
            
            Args: mode: one of the followings (NOTE some ENV does not support all of them)
                - `human`: render to the current display or terminal and
                return nothing. Usually for human consumption.
                - `rgb_array`: Return an numpy.ndarray with shape (x, y, 3),
                representing RGB values for an x-by-y pixel image, suitable
                for turning into a video.
                - `ansi`: Return a string (str) or StringIO.StringIO containing a
                terminal-style text representation. The text can include newlines
                and ANSI escape sequences (e.g. for colors).

            Returns:
                The rendering result, usually an np.ndarray
            """
            return self.env.render(mode)

        def close(self) -> None:
            """ Close the env
            """
            return self.env.close()

        def step(self, actions: Union[torch.Tensor, np.ndarray, list]) -> Tuple[torch.Tensor, torch.Tensor, bool, Any]:
            """ Execute the actions and return the results

            Args: 
                actions: descibing the multi-agent actions. Note the first dimension must represend the number of agents
            Returns: 
                - A `torch.Tensor` for states, whose shape is (agent num, original state shape for EACH agent)
                - A `torch.Tensor` for rewards, of shape (1, agent num)
                - A `bool`, identifying whether an episode is DONE
                - Anything that stores the extra info 
            """
            if not isinstance(actions, torch.Tensor):
                if isinstance(actions, np.ndarray):
                    actions_torch = torch.from_numpy(actions)
                else:
                    actions_torch = torch.Tensor(actions)
            else:
                actions_torch = actions
            # action is of shape(2, 1), varied within (-1.0, 1,0)

            with torch.no_grad():
                actions_int = self.mapper(actions_torch)

            states, rewards, dones, info = self.env.step(actions_int)
            # state: list of Tensor(10,), required Tensor(2,10)
            # rewards: list of int, required Tensor(1,2)
            states = [torch.Tensor(state) for state in states]
            return torch.stack(states), torch.Tensor(rewards).reshape(-1, len(rewards)), any(dones), info
