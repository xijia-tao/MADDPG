import gym
from stable_baselines3.td3.policies import TD3Policy

from typing import Any, Dict, List, Optional, Type, Union
import torch as th
from torch import nn

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

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        #TODO: Return an Actor instance
        pass

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Critic:
        #TODO: Return an Critic instance
        pass

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        #TODO: Feed observation to actor
        pass

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        #TODO: Should be same as _predict
        pass


class Actor():
    
    def __init__(self):
        pass


class Critic(ContinuousCritic):

    def __init__(self):
        pass