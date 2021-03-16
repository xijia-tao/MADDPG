# MADDPG

MADDAPG, it might be more straight to be written as "MA-DDPG", identifying
it as the **multi-agent version** of DDPG. The corresponding ideology was
summarized as "decentralized execution, centralized training." No existing
implementation open-sourced on GitHub were found utilizing the 
[Stable Baseline 3](https://stable-baselines3.readthedocs.io)
(a.k.a. SB3) which wields PyTorch as the AI library. Therefore, we create
this project and aim to implement a robust and adaptable version of MADDPG
with SB3. 

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

### Run the demo

The `main.py` utilize the `Pong-Duel` environment from `ma-gym` which here
serves as the demo. To run it, simply:

```sh
python3 ./main.py
```

_Normally, the interaction of will be renderred directly on the screen. 
If the traning results is expected to be visualized in a video format,
one may need to edit the `maddpg.py` file._

_There is a configuration determining the ways the interactions are
to be presented, namely `EXE_ENV`. Changing it anything else other
than `"TEST"` can instruct the model to save the video frames
as an serialized numpy array file at `output/video.npz`._ 


## Collaboration

For details, you are recommended to go through our 
[Wiki](https://github.com/wwwCielwww/MADDPG/wiki)

_We appologize that currently only the simplified Chinese
version is provided. We promise that the English version
will later be prepared as well._

### Code structure

* `maddpg.py` is the core implementation of the algorithm, which encapsulates a
TD3 object inside and invoke the corresponding TD3 methods for training and
evaluating. 
* `env_wrapper.py` implements the wrapper for multi-agent environments and
the vectorized multi-agent environment which support multiple multi-agent
wrapper in the same time. 
* `ma_policy.py` implements the actor-critic algorithm for multi-agent settings, 
which is corresponding to the core of the paper.
* `buffer.py` customizes the replay buffer for multi-agent interaction. 
* `main.py` implements a demo based on `Pong-Duel` environment from `ma-gym`
* The rest files which are ended with `_test` suffix are unit tests, all based
on the `unittest` module. 

### Switch to other environments

To switch to other environments, this 
[Wiki page](https://github.com/wwwCielwww/MADDPG/wiki#) might 
be helpful. Moreoever, since our test cases are based on
**[MA-GYM](https://github.com/koulanurag/ma-gym)**, it is as
well recommended to check the implementation there and to code
for your own multi-agent environment based on a similar
ideology. 

Breifly speaking, you just need to:
1. Prepare a environment class;
1. Register that class to **GYM** framework with
some name, and finally, 
3. Replace the value of `ENV_NAME` in `main.py`
with the name which you have registered your env with.  
