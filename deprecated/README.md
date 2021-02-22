# A PyTorch Implementation of MADDPG

**Disclaimer:** The code here does NOT strictly comply with the implementation details as suggested in the paper [*Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*](https://arxiv.org/abs/1706.02275). 

### Remark

Since DDPG naturally works with continuous action space, applying the algorithm to discete ones (as most of the envs in ma-gym are) is not so effective. Getting the hyperparameters right is vital for the training as well, which is exactly what I've been doing recently. 

I'll try to write a multi-agent version of PPO as well later. Hope it can give a better result!

## Usage

- To train a new network, run `pong.py`
- To change the default environment, change the line `env = gym.make('PongDuel-v0')` in `pong.py`
- To adjust training arguments, change the line `ddpg(n, agent, env, n_ep=1000, max_t=700, save_every=20)` in `pong.py`
- To load a pretrained network, specify the `load_path` in the form of `['actor_path', 'critic_path']` as an argument of the `Agent` instance in `pong.py`
- By default, an environment is rendered after completing the required number of epsiodes

## Issues with Multi-Agent Environments

- Multi-Agent environments are non-stationary: each agent's policy changes as training progresses.
- Policy-gradient variance grows as number of agents increases.

## MADDPG Key Concepts

- Adopts a framework of
  - **Centralized training**
    - Agents share experience during training;
    - Implemented via a shared experience-replay buffer.
  - **Decentralized execution**
    - Each agent uses only local observations at execution time.
- Extend actor-critic policy gradient methods (e.g. DDPG):
  - Critic is augmented with information about policies of other agents (or, a critic might be programmed to infer those policies);
  - Actor has access only to local information.

## Actor Network

- Actor (policy) network

  - Two layers of 64 neurons each,
  - ReLU activation,
  - Softmax action selection on output layer

  