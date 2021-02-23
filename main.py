import numpy as np
from ma_policy import ma_policy
from maddpg import maddpg

DEFAULT_LEARN_STEPS = 10000
DEFAULT_TEST_STEPS  = 10000

if __name__ == '__main__':
    model = maddpg(ma_policy, None) #TODO
    model.learn(DEFAULT_LEARN_STEPS)

    results = []
    
    for interact in model.execute(DEFAULT_TEST_STEPS):
        if isinstance(interact, np.ndarray):
            results.append(interact)
        elif isinstance(interact, str):
            print(interact)
