from hyper_parameters import training_paramters
import numpy as np
from ma_policy import ma_policy
from maddpg import maddpg



if __name__ == '__main__':
    model = maddpg(ma_policy, "PongDuel-v0")
    model.learn(training_paramters.DEFAULT_LEARN_STEPS)

    results = []
    
    for interact in model.execute(training_paramters.DEFAULT_TEST_STEPS):
        if isinstance(interact, np.ndarray):
            results.append(interact)

    np.savez('output/video.npz', frames=np.array(results))
    
