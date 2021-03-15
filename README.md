# MADDPG

MADDAPG, it might be more straight to be written as "MA-DDPG", identifying
it as the **multi-agent version** of DDPG. The corresponding ideology was
summarized as "decentralized execution, centralized training." No existing
implementation open-sourced on GitHub were found utilizing the 
[Stable Baseline 3](https://stable-baselines3.readthedocs.io)
(a.k.a. SB3) which wields PyTorch as the AI library. Therefore, we create
this project and aim to implement a robust and adaptable version of MADDPG
with SB3. 

### DDPG

The DDPG we used as the base algorithm is actually the 
[TD3 algorithm in SB3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html). 

[TD3](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1802.09477)
is an widely-used extension of DDPG, which uses some tricks to enhance the performance. 

### Multi-agent Concerns

TD3 was single-agent model. Therefore, some tricks are used to tackle the issue, including:

1. Customized both A&C networks;
1. Expand _Q_ which is of shape (Batch Size, 1) to shape (Batch Size, 1, _N_); N 
stands for number of agetns, and
1. Use a wrapper to wrap the MA environment so that it can be trained as if it is 
a single-agent one. 

## Quick start

### Prerequisite

The following modules are required: 
1. PyTorch, instructions of which can be found [here](https://pytorch.org/)
1. MA-GYM (which relies on OpenAI GYM)
    ```py
    pip install gym
    pip install ma-gym
    ```
1. Stable-Baseline-3, which can be installed using: 
    ```py
    pip install stable-baselines3
    ```

### Run the demo env

**_TODO_**

## Collaboration

**_TODO_**

### Code structure

* `maddpg.py` is the core implementation of the algorithm, which encapsulates a
TD3 object inside and invoke the corresponding TD3 methods for training and
evaluating. 
* `env_wrapper.py` implements the wrapper for multi-agent environments and
the vectorized multi-agent environment which support multiple multi-agent
wrapper in the same time. 
* `ma_policy.py` implements the actor-critic algorithm for multi-agent settings, 
which is corresponding to the core of the paper.
* `main.py` implements a demo based on `Pong-Duel` environment from `ma-gym`
* The rest files which are ended with `_test` suffix are unit tests, all based
on the `unittest` module. 

### Switch to other environments

**_TODO_**
