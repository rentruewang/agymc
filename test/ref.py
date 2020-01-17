import argparse

import gym
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--render", action="store_true")
    flags = parser.parse_args()
    num_envs = flags.num_envs
    num_episodes = flags.episodes
    render = flags.render
    envs = tuple(gym.make("CartPole-v1") for _ in range(num_envs))
    for _ in tqdm(range(num_episodes)):
        done = list(False for _ in range(num_envs))
        for (idx, env) in enumerate(envs):
            env.reset()
            while not done[idx]:
                if render:
                    env.render()
                action = env.action_space.sample()
                (_, _, d, _) = env.step(action)
                done[idx] = d
    for env in envs:
        env.close()
