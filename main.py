import numpy as np
from ma_policy import MaPolicy
from maddpg import MaDDPG
from torch import round

DEFAULT_LEARN_STEPS  = 10000
DEFAULT_TEST_STEPS   = 10000
ENV_NAMES            = "PongDuel-v0"
OUTPUT_FRAMES_DIR    = 'output/video.npz'

if __name__ == '__main__':
    model = MaDDPG(
        policy=MaPolicy,
        env=ENV_NAMES,
        mapper=lambda actions: round(actions * 1.5 + 1).flatten().tolist()
    )
    model.learn(DEFAULT_LEARN_STEPS)

    results = []
    
    for interact in model.execute(DEFAULT_TEST_STEPS):
        if isinstance(interact, np.ndarray):
            results.append(interact)

    if len(results) != 0:
        np.savez(OUTPUT_FRAMES_DIR, frames=np.array(results))
    
