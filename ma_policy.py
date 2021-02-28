from hyper_parameters import nn_structure, pong_duel_adapter

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

# A multi-agent policy based on TD3 algorithm
# Paper: https://arxiv.org/abs/1802.09477 (TD3)
# The code conforms to the (official) implementation: https://github.com/sfujim/TD3

class ma_policy(TD3Policy):
    """
    Policy class (with both actor and critic) for MADDPG,
    inherited from TD3Policy.

    For parameters, 
    see https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/td3/policies.py
    :param n_agents: Number of agents in the environment
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        agent_num: int,
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

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None

        self._build()

    def _build(self) -> None:
        self.actor = self.make_actor()
        self.actor_target = deepcopy(self.actor)
        
        self.critic = self.make_critic()
        self.critic_target = deepcopy(self.critic)

    def make_actor(self) -> "Actor":
        return Actor(self.state_dim, self.action_dim, self.agent_num)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> "Critic":
        return Critic(self.state_dim, self.action_dim, self.agent_num)

class Actor(nn.Module):
    """
    Accepts local state information and returns an action
    :param state_dim: dimension of observation space (for a single agent)
    :param action_dim: dimension of action space (for a single agent)
    :param agent_num: number of agents
    """
    def __init__(self, state_dim: int, action_dim: int, agent_num: int):
        super(Actor, self).__init__()

        self._nn_list = [] # list of agents, _nn_list[i] => actor for agent[i]

        for _ in range(agent_num):
            l1 = nn.Linear(state_dim, nn_structure.ACTOR_FC_NODES)
            l2 = nn.Linear(nn_structure.ACTOR_FC_NODES, nn_structure.ACTOR_FC_NODES)
            l3 = nn.Linear(nn_structure.ACTOR_FC_NODES, action_dim)
            self._nn_list.append((l1, l2, l3))

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
    def __init__(self, state_dim, action_dim, n_agents):
        super(Critic, self).__init__()
        self._qnn_lst = []

        for _ in range(n_agents):
            l1 = nn.Linear((state_dim + action_dim) * n_agents, nn_structure.CRITIC_FC_NODES)
            l2 = nn.Linear(nn_structure.CRITIC_FC_NODES, nn_structure.CRITIC_FC_NODES)
            l3 = nn.Linear(nn_structure.CRITIC_FC_NODES, 1)
            self._qnn_lst.append((l1, l2, l3))

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

    # def Q1(self, state, action):
    #     sa = torch.cat([state, action], 1)
        
    #     q1 = F.relu(self.l1(sa))
    #     q1 = F.relu(self.l2(q1))
    #     q1 = self.l3(q1)
    #     return q1