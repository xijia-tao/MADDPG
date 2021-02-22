import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import *

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 512
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
WEIGHT_DECAY = 0.0001
UPDATE_EVERY = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():

	def __init__(self, s_dim, a_dim, seed, load_path):
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.seed = random.seed(seed)

		self.actor_local = Actor(s_dim, a_dim, seed).to(device)
		self.actor_target = Actor(s_dim, a_dim, seed).to(device)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

		self.critic_local = Critic(s_dim, a_dim, seed).to(device)
		self.critic_target = Critic(s_dim, a_dim, seed).to(device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

		self.noise = OUNoise(a_dim, seed)

		self.memory = ReplayBuffer(a_dim, BUFFER_SIZE, BATCH_SIZE, seed)

		self.update_every = UPDATE_EVERY

		self.t_step = 0

		if load_path:
			self.actor_local.load_state_dict(torch.load(load_path[0]))
			self.critic_local.load_state_dict(torch.load(load_path[1]))

	def step(self, s, a, r, s_, done):
		self.memory.add(s, a, r, s_, done)

		self.t_step = (self.t_step + 1) % self.update_every

		if self.t_step == 0:
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.sample()
				self.learn(experiences, GAMMA)

	# Further Inspection Required
	def act(self, s, add_noise=True):
		s = torch.from_numpy(s).float().to(device)
		self.actor_local.eval()
		with torch.no_grad():
			action = self.actor_local(s).cpu().data.numpy()
		self.actor_local.train()
		if add_noise:
			action += self.noise.sample()
		return action

	def reset(self):
		self.noise.reset()

	def learn(self, experiences, gamma):
		s, a, r, s_, dones = experiences
		
		# Update Critic
		a_ = self.actor_target(s_)

		q_targets_ = self.critic_target(s_, a_)
		q_targets = r + (gamma * q_targets_ * (1 - dones))
		q_expected = self.critic_local(s, a)

		critic_loss = F.mse_loss(q_expected, q_targets)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Update Actor
		a_pred = self.actor_local(s)
		actor_loss = -self.critic_local(s, a_pred).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update Target Networks
		self.soft_update(self.critic_local, self.critic_target, TAU)
		self.soft_update(self.actor_local, self.actor_target, TAU)

	def soft_update(self, local, target, tau):
		for tar_param, loc_param in zip(target.parameters(), local.parameters()):
			tar_param.data.copy_(tau * loc_param + (1.0 - tau) * tar_param)


class OUNoise:

	def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
		self.mu = mu * np.ones(size)
		self.theta = theta
		self.sigma = sigma
		self.seed = random.seed(seed)
		self.reset()

	def reset(self):
		self.state = copy.copy(self.mu)

	def sample(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
		self.state = x + dx
		return self.state


class ReplayBuffer:

	def __init__(self, a_dim, buffer_size, batch_size, seed):
		self.a_dim = a_dim
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["s", "a", "r", "s_", "done"])
		self.seed = random.seed(seed)

	def add(self, s, a, r, s_, done):
		e = self.experience(s, a, r, s_, done)
		self.memory.append(e)

	def sample(self):
		experiences = random.sample(self.memory, k=self.batch_size)

		s = torch.from_numpy(np.vstack([e.s for e in experiences if e is not None])).float().to(device)
		a = torch.from_numpy(np.vstack([e.a for e in experiences if e is not None])).float().to(device)
		r = torch.from_numpy(np.vstack([e.r for e in experiences if e is not None])).float().to(device)
		s_ = torch.from_numpy(np.vstack([e.s_ for e in experiences if e is not None])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

		return (s, a, r, s_, dones)

	def __len__(self):
		return len(self.memory)

