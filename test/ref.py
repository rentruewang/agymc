import argparse

import gym
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--episodes", type=int)
    flags = parser.parse_args()
    num_envs = flags.num_envs
    num_episodes = flags.episodes
    envs = tuple(gym.make("CartPole-v0") for _ in range(num_envs))
    for _ in tqdm(range(num_episodes)):
        done = list(False for _ in range(num_envs))
        for (idx, env) in enumerate(envs):
            env.reset()
            while not done[idx]:
                env.render()
                action = env.action_space.sample()
                (_, _, d, _) = env.step(action)
                done[idx] = d
    for env in envs:
        env.close()
