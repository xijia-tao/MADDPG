# A PyTorch Implementation of MADDPG

**Disclaimer:** The code here does NOT strictly comply with the implementation details as suggested in the paper [*Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*](https://arxiv.org/abs/1706.02275). 

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

  