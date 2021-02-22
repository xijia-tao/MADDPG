import gym
import ma_gym
import numpy as np

from agent import Agent
from algorithm import ddpg
from render import render
env = gym.make('PongDuel-v0')

n = len(env.action_space)
a_dim = env.action_space[0].n
s_dim = env.observation_space[0].shape[0]
render_after = True

agent = Agent(s_dim=s_dim, a_dim=a_dim, seed=2, load_path=None)

ddpg(n, agent, env, n_ep=1000, max_t=700, save_every=20)

if render_after:
	render(n, agent, env)
