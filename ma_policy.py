from hyper_parameters import nn_structure
from utils import pong_duel_adapter

import torch
from torch import nn
import torch.nn.functional as F
from stable_baselines3.td3.policies import TD3Policy
import gym
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
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
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
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
        state_dim: int = 10,
        action_dim: int = 1,
        agent_num: int = 2,
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
            share_features_extractor,
        )
        self.actor_kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "agent_num": agent_num
        }
        self.critic_kwargs = self.actor_kwargs.copy()

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None

        self.build(lr_schedule)

    def build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor_()
        self.actor_target = deepcopy(self.actor)
        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1))
        
        self.critic = self.make_critic_()
        self.critic_target = deepcopy(self.critic)
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1))

    def make_actor_(self) -> "Actor":
        return Actor(**self.actor_kwargs)

    def make_critic_(self) -> "Critic":
        return Critic(**self.critic_kwargs)


class Actor(nn.Module):
    """
    Accepts local state information and returns an action
    :param state_dim: dimension of observation space (for a single agent)
    :param action_dim: dimension of action space (for a single agent)
    :param agent_num: number of agents
    """
    def __init__(self, state_dim: int, action_dim: int, agent_num: int):
        super(Actor, self).__init__()

        self._nn_list = nn.ModuleList() # list of agents, _nn_list[i] => actor for agent[i]

        for _ in range(agent_num):
            actor_i = nn.ModuleList()
            l1 = nn.Linear(state_dim, ACTOR_FC_NODES)
            l2 = nn.Linear(ACTOR_FC_NODES, ACTOR_FC_NODES)
            l3 = nn.Linear(ACTOR_FC_NODES, action_dim)
            actor_i.extend([l1, l2, l3])
            self._nn_list.append(actor_i)

    def forward(self, state):
        action_result = []
        for layers in self._nn_list:
            a = F.relu(layers[0](state))
            a = F.relu(layers[1](a))
            action_result.append(torch.tanh(layers[2](a)))
        
        return pong_duel_adapter(action_result) #TODO: test it


class Critic(nn.Module):
    """
    Accepts global information (state & action) 
    and returns 2 Q values that rate the action
    :param state_dim: dimension of observation space (for a single agent)
    :param action_dim: dimension of action space (for a single agent)

    Remark:
    - The design of multiple Q functions conforms to TD3 algorithm.
    This can increase robustness of our policy.
    - A critic is only needed during training.
    """
    def __init__(self, state_dim, action_dim, agent_num):
        super(Critic, self).__init__()

        self._qnn_lst = nn.ModuleList()

        for _ in range(agent_num):
            critic_i = nn.ModuleList()
            l1 = nn.Linear((state_dim + action_dim) * agent_num, CRITIC_FC_NODES)
            l2 = nn.Linear(CRITIC_FC_NODES, CRITIC_FC_NODES)
            l3 = nn.Linear(CRITIC_FC_NODES, 1)
            critic_i.extend([l1, l2, l3])
            self._qnn_lst.append(critic_i)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        qs = []
        for eachNN in self._qnn_lst:
            eachQ = F.relu(eachNN[0](sa))
            eachQ = F.relu(eachNN[1](eachQ))
            eachQ = eachNN[2](eachQ)
            qs.append(eachQ)
        # qs: n * (batch_size, 1)
        return torch.cat(qs, 1)