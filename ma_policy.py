from hyper_parameters import nn_structure, pong_duel_adapter
import gym
from stable_baselines3.td3.policies import TD3Policy

from typing import Any, Dict, List, Optional, Type, Union
import torch
import torch as th
from torch import nn
import torch.nn.functional as F

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

    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule: Schedule,
                 net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 n_critics: int = 2,
                 share_features_extractor: bool = True,
                 n_agents: int = 2,
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
        self.n_agents = n_agents

        #TODO: Design network architecture, args
        #TODO: Initialize AC & their target networks

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        #TODO: Create AC & their target networks
        pass

    def _get_data(self) -> Dict[str, Any]:
        #TODO: Reimplement / leave as it is
        pass

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> "Actor":
        #TODO: Return an Actor instance
        pass

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> "Critic":
        #TODO: Return an Critic instance
        pass

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        #TODO: Feed observation to actor
        pass

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        #TODO: Should be same as _predict
        pass


class Actor(nn.Module):
    """
    Accepts local state information and returns an action
    :param state_dim: dimension of observation space (for a single agent)
    :param action_dim: dimension of action space (for a single agent)
    :param agent_num: number of agents
    """
    
    def __init__(self, state_dim: int, action_dim: int, agent_num: int):
        super(Actor, self).__init__()

        self._nn_list   = [] # list of agents, _nn_list[i] => actor for agent[i]

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
            action_result.append(th.tanh(layers[2](a)))
        
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

        # Q1 architecture
        self.l1 = nn.Linear((state_dim + action_dim) * n_agents, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear((state_dim + action_dim) * n_agents, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = th.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1