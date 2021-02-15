import gym
import ma_gym
import numpy as np
from algorithm import DDPG

MAX_EPISODES = 2000
MAX_EP_STEPS = 2000
MEMORY_CAPACITY = 3000
RENDER = False
ENV_NAME = "PongDuel-v0"


env = gym.make(ENV_NAME)
env = env.unwrapped

# Continuous Observation & Discrete Action Space
s_dim = env.observation_space[0].shape[0]
a_dim = env.action_space[0].n
n = len(env.observation_space)

ddpg = DDPG(s_dim, a_dim, n)

# Only one agent acts in one epsiode
for i in range(MAX_EPISODES):

	s = env.reset()
	ep_reward = [0] * n
	done = [0] * n

	while not all(done):

		if RENDER:
			env.render()

		act, prob = ddpg.get_action(s, i%n)
		a = [act if j == i%n else 0 for j in range(n)]
		s_, r, done, _ = env.step(a)

		ddpg.store_mem(s, a, r, s_, i%n)

		if ddpg.pointer > MEMORY_CAPACITY:
			ddpg.learn(i%n)

		s = s_

		ep_reward_ = []
		for rew, rew_ in zip(r, ep_reward):
			ep_reward_.append(rew + rew_)
		ep_reward = ep_reward_


		print('Episode:', i, ' Reward for')
	    for k in range(n):
	        print(f'Agent {k}: {ep_reward[k]}', end='\t')
	    print()
	    print(f'{sum(ep_reward)}')
