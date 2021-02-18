import gym
import ma_gym
import numpy as np

from agent import Agent
from algorithm import ddpg

env = gym.make('PongDuel-v0')

n = len(env.action_space)
a_dim = env.action_space[0].n
s_dim = env.observation_space[0].shape[0]

agent = Agent(s_dim=s_dim, a_dim=a_dim, seed=2)

ddpg(n, agent, env)
