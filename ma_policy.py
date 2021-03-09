import enum
from os import stat
import numpy as np
from stable_baselines3.common.policies import BaseModel, BasePolicy, ContinuousCritic
from torch import tensor
from deprecated.model import Critic

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from stable_baselines3.td3.policies import TD3Policy, Actor as single_actor
from gym import spaces
from typing import Any, Callable, Dict, List, Optional, Type, Union

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule

ACTOR_FC_NODES  = 256
CRITIC_FC_NODES = 256

# A multi-agent policy based on TD3 algorithm
# Paper: https://arxiv.org/abs/1802.09477 (TD3)
# The code conforms to the (official) implementation: https://github.com/sfujim/TD3


class ma_policy(TD3Policy):
    """
    Policy class (with both actor and critic) for MADDPG,
    inherited from TD3Policy.

    For parameters, 
    see https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/policies.py
    
    :param state_dim: dimension of observation space (for a single agent)
    :param action_dim: dimension of action space (for a single agent)
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
        self.actor_kwargs = self.actor_kwargs.update(extra_kwarg_)
        self.critic_kwargs = self.critic_kwargs.update(extra_kwarg_)

        super()._build(lr_schedule)

    def _build(self, lr_schedule: Callable) -> None:
        # the _build method will be called in the super().__init__() method
        # when the extra argument has not been set up yet
        # we must defer the invoking until the very end of the __init__()
        # for this class
        pass

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> "ma_actor":
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ma_actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> "ma_critic":
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ma_critic(**critic_kwargs).to(self.device)

class ma_actor(BasePolicy):
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
        super(ma_actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )
        real_observation_space = spaces.Box(observation_space.low, observation_space.high,observation_space.shape[1:])
        real_action_space      = spaces.Box(action_space.low, action_space.high, action_space[1:])

        self._agents = [single_actor(real_observation_space, real_action_space, net_arch, features_extractor, features_dim, activation_fn, normalize_images) for _ in range(agent_num)]

        for idx, agent_model in enumerate(self._agents):
            self.add_module(f"ag_{idx}", agent_model)
    
    def forward(self, obs: Tensor, deterministic: bool = True) -> Tensor:
        results = []
        for i,agent in enumerate(self._agents):
            results.append(agent.forward(obs[:,i], deterministic)) # first dim is the batch size
        return torch.stack(results)

    def _predict(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        return self.forward(obs, deterministic)
    
    def _get_data(self) -> Dict[str, Any]:
        assert len(self._agents) != 0
        sub_data = self._agents[0]._get_data()
        sub_data.update({
            "agent_num": len(self._agents)
        })
        return sub_data
        

class ma_critic(BaseModel):
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
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )
        real_observation_space = spaces.Box(observation_space.low, observation_space.high,observation_space.shape[1:])

        self._critics = [ContinuousCritic(real_observation_space, action_space, net_arch, features_extractor, features_dim, activation_fn, normalize_images, n_critics, share_features_extractor) for _ in range(agent_num)]
        for idx, criticNN in enumerate(self._critics):
            self.add_module(f"ag_c_{idx}", criticNN)

    def forward(self, all_obs: Tensor, all_actions: Tensor):
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
        results = [critic.forward(all_obs[:,i], all_actions) for i,critic in enumerate(self._critics)] # first dim is the batch size
        # results is # assume agent_num = 3, num_critic = 2
        # [
        #    critic for ag_1: (Batch,1)
        #    critic for ag_2: (Batch,1)
        #    critic for ag_3: (Batch,1)
        # ]
        # what we need is (Batch, 1, 3)
        return torch.cat(results, dim = 1).reshape(-1,1,len(results))

