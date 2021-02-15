import torch
from torch import nn
import torch.nn.functional as F


class Actor(nn.Module):
	"""
	Initial Attempt:
	Actor is only aware of its own states.
	A properly working instance should take in a tensor as states,
	and output the probablity of each possible action.
	"""
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, a_dim)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
	"""
	Initial Attempt:
	Critic is only aware of its agent's states.
	A properly working instance should take in 2 tensors as states and actions,
	and output the estimated Q value under that specific situation.
	"""
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc2 = nn.Linear(128 + a_dim, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, s, a):
        x = F.relu(self.fc1(s))
        combined = torch.cat([x, a], 1)
        x = F.relu(self.fc2(combined))
        return self.fc4(F.relu(self.fc3(x)))