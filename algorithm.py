import torch 
import numpy as np
from model import *
import random

# Hyperparams (subject to change)
LR_A = 0.01
LR_C = 0.01
TAU = 0.01
GAMMA = 0.9
BATCH_SIZE = 64
MEMORY_CAPACITY = 3000
EPSILON = 0.5

class DDPG(object):
	def __init__(self, s_dim, a_dim, n):
		pass

	def get_action(self, s, i):
		pass

	def learn(self, i):
		pass

	def store_mem(self, s, a, r, s_):
		pass