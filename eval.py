import numpy as np
import matplotlib.pyplot as plt

def eval(model, ep):
    env = model._env
    all_episode_rewards = []
    for i in range(ep):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(np.array(reward).flatten().tolist())
        episode_rewards = np.array(episode_rewards)
        all_episode_rewards.append([sum(episode_rewards[::, i]) for i in range(env.agent_num())])
    
    plt.plot(range(1, ep+1), [rew[0] for rew in all_episode_rewards])
    plt.plot(range(1, ep+1), [rew[1] for rew in all_episode_rewards])
    plt.savefig('eval.jpg')