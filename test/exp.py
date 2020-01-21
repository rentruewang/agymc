import argparse
import time

import agymc

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

    envs = agymc.make("CartPole-v0", num_envs)
    if verbose:
        import tqdm

        iterable = tqdm.tqdm(range(num_episodes))
    else:
        iterable = range(num_episodes)
    for _ in iterable:
        done = list(False for _ in range(num_envs))
        envs.reset()
        while not all(done):
            if render:
                envs.render()
            action = envs.action_space.sample()
            # using time.sleep to simulate workflow
            # time.sleep blocks the current thread
            # however we wrapped the environment in a rather nice way
            # such that concurrency still applies
            # the result: It won't block.
            _ = envs.parallel(lambda *args: time.sleep, [num_envs * [1]])
            (_, _, done, _) = envs.step(action)
    envs.close()
