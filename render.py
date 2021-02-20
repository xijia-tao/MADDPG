import time
import numpy as np


def render(n, agent, env):
	s = np.array(env.reset())
	agent.reset()
	while True:
		
		a = []
		for i_ in range(n):
			a.append(agent.act(np.array(s[i_])))
			agent.reset()
		a = np.array(a)
		s_, r, dones, _ = env.step([np.argmax(a_) for a_ in a])
		for i_ in range(n):
			agent.step(s[i_], a[i_], r[i_], s_[i_], dones[i_])

		s = s_

		env.render()
		time.sleep(0.1)

		if np.any(dones):
			break