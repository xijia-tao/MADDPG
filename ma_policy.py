from stable_baselines3.common.policies import BaseModel, BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

import torch
from torch import nn, Tensor
from stable_baselines3.td3.policies import TD3Policy, Actor as single_actor
from gym import spaces
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Type, Union

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule

# A multi-agent policy based on TD3 algorithm
# Paper: https://arxiv.org/abs/1802.09477 (TD3)
# The code conforms to the (official) implementation: https://github.com/sfujim/TD3


class ma_policy(TD3Policy):
    """
    Policy class (with both actor and critic) for MADDPG,
    inherited from TD3Policy.

    For parameters, 
    see https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/policies.py
    
    :param agent_num: number of agents
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        agent_num: int = 2
        ):

        super(ma_policy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        extra_kwarg_ = {
            "agent_num": agent_num
        }
        self.actor_kwargs.update(extra_kwarg_)
        self.critic_kwargs.update(extra_kwarg_)

        super()._build(lr_schedule)

    def _build(self, lr_schedule: Callable) -> None:
        # the _build method will be called in the super().__init__() method
        # when the extra argument has not been set up yet
        # we must defer the invoking until the very end of the __init__()
        # for this class
        pass

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> "ma_actor":
        """ Create an actor network

        The function will be invoked within super()._build(...) to prepare the actor for the TD3.

        Args: 
            features_extractor: network layouts for feature extration
        
        Returns:
            An ma_actor, which independently executes multiple actors simultaneously
        """
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ma_actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> "ma_critic":
        """ Createa the critic network

        The function will be invoked within super()._build(...) TWICE: one for creating the Q network
        and the other for creating the Q-target.

        Args: 
            features_extractor: network layouts for feature extration
            
        Returns:
            An ma_critic, which runs mutiple Q network simultaneously to create a critic value for 
            EACH agent. 
        """
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ma_critic(**critic_kwargs).to(self.device)


class ma_actor(BasePolicy):
    """ The multi-agent actor class. 

    The class manipulates one network (which is a td3.policies.Actor object) in the same time, 
    each of which genearting the action for one corresponding agent. The output are concatenated
    together to match the required input from the environment. 
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        agent_num: int = 1
    ) -> None:
        """ Constructor

        Args: 
            observation_space: Obervation space
            action_space: Action space
            net_arch: Network architecture
            features_extractor: Network to extract features
                (a CNN when using images, a nn.Flatten() layer otherwise)
            features_dim: Number of features
            activation_fn: Activation function
            normalize_images: Whether to normalize images or not, 
                dividing by 255.0 (True by default)
            agent_num: the number of agents
        """
        super(ma_actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        obs_low = np.array(observation_space.low).flatten()[0]
        obs_high = np.array(observation_space.high).flatten()[0]
        act_low = np.array(action_space.low).flatten()[0]
        act_high = np.array(action_space.high).flatten()[0]
        
        real_observation_space = spaces.Box(obs_low, obs_high, observation_space.shape[1:])
        real_action_space      = spaces.Box(act_low, act_high, (1,) + action_space.shape[1:])
        # origin feature dim is calculated based on observation_space, instead of the real one
        features_dim = get_flattened_obs_dim(real_observation_space)

        self._agents = [single_actor(real_observation_space, real_action_space, net_arch, features_extractor, features_dim, activation_fn, normalize_images) for _ in range(agent_num)]

        for idx, agent_model in enumerate(self._agents):
            self.add_module(f"ag_{idx}", agent_model)
    
    def forward(self, obs: Tensor, deterministic: bool = True) -> Tensor:
        """ Forwarding the network

        Args: 
            obs: the observation from the env wrapper, of shape: 
                (Batch Size, Num Agent, Observation Dim...)
            deterministic: whether the policy should be determinisitic

        Returns:
            The actions for all agents, of shape:
            (Batch Size, Num Agent, Action Dim...)
        """
        results = []
        for i,agent in enumerate(self._agents):
            results.append(agent.forward(obs[:,i])) # first dim is the batch size
        # results if of list of Tensor of shape(Batch Size, Action Dim...)
        batch_size = obs.shape[0]
        num_agent  = len(results)
        return torch.cat(results, dim=1).reshape(batch_size, num_agent, -1)

    def _predict(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """ Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        Args:
            observation:
            deterministic: Whether to use stochastic or deterministic actions
            Taken action according to the policy
        """
        return self.forward(obs, deterministic)
    
    def _get_data(self) -> Dict[str, Any]:
        """ Get data that need to be saved in order to re-create the model.
        
        This corresponds to the arguments of the constructor.

        Returns:
            A dict for parameters to be stored
        """
        assert len(self._agents) != 0
        sub_data = self._agents[0]._get_data()
        sub_data.update({
            "agent_num": len(self._agents)
        })
        return sub_data
        

class ma_critic(BaseModel):
    """ The multi-agent critic class.

    Double Q-learning for multiple-agent scenario. The internal implementation 
    for each agent is a ContinuousCritic object
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        agent_num: int = 1
    ):
        """ Constructor

        Args: 
        	observation_space: Obervation space
        	action_space: Action space
        	net_arch: Network architecture
        	features_extractor: Network to extract features
               (a CNN when using images, a nn.Flatten() layer otherwise)
        	features_dim: Number of features
        	activation_fn: Activation function
        	normalize_images: Whether to normalize images or not,
                dividing by 255.0 (True by default)
        	n_critics: Number of critic networks to create.
        	share_features_extractor: Whether the features extractor is shared or not
                between the actor and the critic (this saves computation time)
        """
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        obs_low = np.array(observation_space.low).flatten()[0]
        obs_high = np.array(observation_space.high).flatten()[0]
        real_observation_space = spaces.Box(obs_low, obs_high, observation_space.shape[1:])

        self._critics = [ContinuousCritic(real_observation_space, action_space, net_arch, features_extractor, features_dim, activation_fn, normalize_images, n_critics, share_features_extractor) for _ in range(agent_num)]
        for idx, criticNN in enumerate(self._critics):
            self.add_module(f"ag_c_{idx}", criticNN)

    def forward(self, all_obs: Tensor, all_actions: Tensor):
        """ Forwarding the network

        Args: 
            all_obs: the observation from the env wrapper, of shape: 
                (Batch Size, Num Agent, Observation Dim...)
            all_actions: The actions for all agents, of shape:
                (Batch Size, Num Agent, Action Dim...)

        Returns:
            A list of critics, each of shape: (Batch Size, 1, Num agent)
            The length of the list matches the n_critics
        """
        results = [critic.forward(all_obs[:,i], all_actions) for i,critic in enumerate(self._critics)] # first dim is the batch size
        # results is # assume agent_num = 3, num_critic = 2
        # [
        #    critic for ag_1: [(Batch, 1), (Batch, 1)]
        #    critic for ag_2: [(Batch, 1), (Batch, 1)]
        #    critic for ag_3: [(Batch, 1), (Batch, 1)]
        # ]
        # what we need is [(Batch, 1, 3), (Batch, 1, 3)]
        m, n = len(results[0]), len(results)
        stacked_results = []
        for j in range(m):
            stacked_results.append(
                torch.cat([results[i][j] for i in range(n)], dim=1).reshape(-1,1,n) # batch size, 1, agent_num 
            )
        return stacked_results

    def q1_forward(self, all_obs: Tensor, all_actions: Tensor) -> Tensor:
        """ Only predict the Q-value using the first network.

        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).

        Args: 
            all_obs: the observation from the env wrapper, of shape: 
                (Batch Size, Num Agent, Observation Dim...)
            all_actions: The actions for all agents, of shape:
                (Batch Size, Num Agent, Action Dim...)

        Returns: 
            A critic tensor of shape (Batch size, 1, Num Agent)
        """
        results = [critic.q1_forward(all_obs[:,i], all_actions) for i,critic in enumerate(self._critics)] # first dim is the batch size
        # results is # assume agent_num = 3, num_critic = 2
        # [
        #    critic for ag_1: (Batch,1)
        #    critic for ag_2: (Batch,1)
        #    critic for ag_3: (Batch,1)
        # ]
        # what we need is (Batch, 1, 3)
        return torch.cat(results, dim = 1).reshape(-1,1,len(results))

