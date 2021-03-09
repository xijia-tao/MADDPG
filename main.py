import numpy as np
from ma_policy import ma_policy
from maddpg import maddpg

DEFAULT_LEARN_STEPS  = 10000
DEFAULT_TEST_STEPS   = 10000
ENV_NAMES            = "PongDuel-v0"

if __name__ == '__main__':
    model = maddpg(ma_policy, ENV_NAMES)
    model.learn(DEFAULT_LEARN_STEPS)

    results = []
    
    for interact in model.execute(DEFAULT_TEST_STEPS):
        if isinstance(interact, np.ndarray):
            results.append(interact)

    if len(results) != 0:
        np.savez('output/video.npz', frames=np.array(results))
    
