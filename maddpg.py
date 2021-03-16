# from hyper_parameters import td3_parameters, env_wrapper
from utils.buffer import MultiAgentReplayBuffer
from utils.env_wrapper import VectorizedMultiAgentEnvWrapper
import gym
import numpy as np

from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Type, Union
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import TD3Policy

EXE_ENV = 'TEST'

LEARNING_RATE       = 0.001
BUFFER_SIZE         = 100000
LEARNING_STARTS     = 100
BATCH_SIZE          = 100
TAU                 = 0.005
GAMMA               = 0.99
GRADIENT_STEPS      = -1 #DISABLE
N_EPISODES_ROLLOUT  = 1
POLICY_DELAY        = 2

class agents: 
    """ 
    A iterator, successively executing the trained model in the given environment

    This class is implemented as an ITERATOR. It will iterate until the agents in the environment 
    complete their iteration thoroughouly. For each iteration step, it utilizes the input model
    to predict the desirable action, iteracts with the environment using that action and return
    the action results renderred by the environment. 

    Args: 
        model:    the trained MADDPG model. 
        min_step: once the number of iteractions exceeds the min_step AND the environment is completed
                  step the execution
        max_step: once the number of iteractions exceeds the max_step, terminate IMMEDIATELY, 
                  regardless the current status of environment. If not set or set to a negative
                  value, this mechanism will not be in effect. 
    """
    def __init__(self,
        model: "MaDDPG",
        min_steps: int,
        max_steps: int = -1) -> None:
        self.__model     = model
        self.__env       = self.__model.env
        self.__obs       = self.__env.reset()
        self.__min_steps = min_steps
        self.__max_steps = max_steps
        self.__count     = 0
        self.__next_time_terminate = False


    def _iteract(self) -> Tuple[Any, bool]: 
        """ 
        Predict for the next action, and then with the action interact with the environment

        Returns: 
            The result after the iteraction, and
            A boolean to spedicify whether the environment is completed
        """
        action, _ = self.__model.predict(self.__obs)
        self.__obs, _, done, _ = self.__env.step(action)
        self.__env.render(mode = 'human' if EXE_ENV == 'TEST' else 'rgb_array'), done

    def __next__(self) -> Any:
        if self.__next_time_terminate or self.__count >= self.__max_steps > 0:
            raise StopIteration
        render_result, done = self._iteract()
        self.__count += 1
        if done: 
            self.__obs = self.__env.reset()
            if self.__count >= self.__min_steps: 
                self.__next_time_terminate = True
        return render_result
        

    def __iter__(self) -> "agents":
        self.__obs   = self.__env.reset()
        self.__count = 0
        self.__next_time_terminate = False
        return self

class MaDDPG:
    """ 
    Implementation of Multi-agent DDPG (MADDPG)

    The original paper is available at https://arxiv.org/abs/1706.02275v4
    Here, instead of following the existing implementation at https://github.com/openai/maddpg/tree/master/maddpg
    which is basde on TensorFlow, we re-implement the model utilizing Stable-Baseline 3, constructed
    on PyTorch. Meanwhile, the underlying DDPG algorithm are replaced by TD3. 

    Args: 
        policy: the actor-critic policy, inherited from TD3Policy, or a string if the policy 
                has been registered
        env:    name of the environment
        mapper: a function that converts the action outputs from the model to what can be
                accepted by the env object. See VectorizedMultiAgentEnvWrapper.MultiAgentEnvWrapper
                for details. 
    """
    def __init__(self, 
        policy: Union[str, Type[TD3Policy]],
        env: str, 
        mapper: Callable[[Tensor], Union[Tensor, np.ndarray, list]] = None
    ) -> None:
        vecEnv = self._get_env(env, mapper)

        self._env       = vecEnv
        self._policy    = policy
        self._ddpg      = TD3(
            self._policy,
            self._env,
            learning_rate   = LEARNING_RATE,
            buffer_size     = BUFFER_SIZE,
            learning_starts = LEARNING_STARTS,
            batch_size      = BATCH_SIZE,
            tau             = TAU,
            gamma           = GAMMA,
            policy_delay    = POLICY_DELAY,
            train_freq      = (N_EPISODES_ROLLOUT, 'episode'),
            policy_kwargs   = {"agent_num": vecEnv.agent_num()}
        )
        self._ddpg.replay_buffer = MultiAgentReplayBuffer(
            buffer_size       = BUFFER_SIZE, 
            observation_space = vecEnv.observation_space,
            action_space      = vecEnv.action_space,
            device            = self._ddpg.replay_buffer.device,
            n_envs            = len(vecEnv.envs),
            n_agent           = vecEnv.agent_num, 
            optimize_memory_usage = self._ddpg.replay_buffer.optimize_memory_usage
        )
            
    def learn(self, total_timesteps = 10000) -> None:
        """ 
        Learn from the environment using the policy

        Args
            total_timesteps: learn for how many timesteps
        """
        self._ddpg.learn(total_timesteps)

    @staticmethod
    def _get_env(env: Union[str, gym.Env], mapper: Callable[[Tensor], Union[Tensor, np.ndarray, list]] = None):
        """ 
        Get & wrap the underlying environment for the model

        Args: 
            env: the environment, string if the environment has been registered
            mapper: the function converting actions from different spaces, see
                VectorizedMultiAgentEnvWrapper.MultiAgentEnvWrapper for details

        Returns: 
            A wrapper of ma-gym environment that is compatible 
            with the stable-baselines3 algorithms
        """
        return VectorizedMultiAgentEnvWrapper([(lambda: env, mapper)])

    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        Args
            observation:    The input observation

        Returns
            The model's action and the next state
            (used in recurrent policies)
        """
        return self._ddpg.predict(observation, deterministic=True)

    def execute(self, num_of_step: int) -> agents:
        """ 
        Execute the policy in the environment
        
        Get a iterator over the environment, which continuously instructs all the agents to
        interacts with the environment. 

        Args: 
            num_of_step: stop the execution after how many steps

        Returns: 
            An iterator over the agent
        """
        return iter(agents(self, self._env, num_of_step, max_steps=2*num_of_step))