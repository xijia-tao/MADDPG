import random
import torch
import numpy as np
from collections import deque


def ddpg(n, agent, env, n_ep, max_t, save_every):
	scores_deque = deque(maxlen=100)
	scores_list = []
	max_score = -np.Inf

	for i in range(1, n_ep+1):

		s = np.array(env.reset())
		agent.reset()
		scores = np.zeros(n)
		for t in range(max_t):
			
			a = []
			for i_ in range(n):
				a.append(agent.act(np.array(s[i_])))
				agent.reset()
			a = np.array(a)
			s_, r, dones, _ = env.step([np.argmax(a_) for a_ in a])
			for i_ in range(n):
				agent.step(s[i_], a[i_], r[i_], s_[i_], dones[i_])
			
			scores += r
			s = s_
			if np.any(dones):
				break
		score = np.max(scores)
		scores_deque.append(score)
		scores_list.append(score)

		if i % save_every == 0:
			torch.save(agent.actor_local.state_dict(), f'ckpt_actor_{i}.pth')
			torch.save(agent.critic_local.state_dict(), f'ckpt_critic_{i}.pth')

		print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i, np.mean(scores_deque), score))
	
	return scores_list