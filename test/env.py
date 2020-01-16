from agymc.src.env import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--episodes", type=int)
    flags = parser.parse_args()
    num_envs = flags.num_envs
    num_episodes = flags.episodes
    envs = make("CartPole-v0", num_envs)
    # raise SystemExit
    for _ in tqdm(range(num_episodes)):
        # print(_)
        done = list(False for _ in range(num_envs))
        envs.reset()
        while not all(done):
            envs.render()
            action = envs.action_space.sample()
            (_, _, done, _) = envs.step(action)
            # (obs, rew, done, _) = envs.step(action)
            # print(obs)
            # print(rew)
            # print(done)
            # print()
    envs.close()
