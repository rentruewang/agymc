import argparse

from agymc import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    flags = parser.parse_args()
    num_envs = flags.num_envs
    num_episodes = flags.episodes
    render = flags.render
    verbose = flags.verbose
    envs = make("CartPole-v1", num_envs)
    if verbose:
        import tqdm

        iterable = tqdm(range(num_episodes))
    else:
        iterable = range(num_episodes)
    for _ in iterable:
        done = list(False for _ in range(num_envs))
        envs.reset()
        while not all(done):
            if render:
                envs.render()
            action = envs.action_space.sample()
            (_, _, done, _) = envs.step(action)
    envs.close()
