# agymc

![gym](./assets/gym.png)

**For reinforcement learning and concurrency lovers out there ...**

### TL;DR

- Mostly the same API as gym, except now multiple environments are run.
- Envs are run concurrently, which means speedup with time consuming operations such as backprop, render etc..

### Intro

This is a concurrent wrapper for OpenAI Gym library that runs multiple environments concurrently, which means running faster in training\* without consuming more CPU power.

### What exactly is _concurrency_ ?

Maybe you have heard of _parallel computing_ ? When we say we execute things in parallel, we run the program on _multiple_ processors, which offers significant speedup. _Concurrency computing_ has a broader meaning, though. The definition of a _concurrent_ program, is that it is designed not to execute sequentially, and will one day be executed parallelly\*\*. A _concurrent program_ can run on a sigle processor or multiple processors. These tasks may communicate with each other, but have separate private states hidden from others.

### Can _concurrency_ be applied on a single processor ?

Yes, _concurrency_ means splitting the program into smaller subprograms, allowing some parts of code to be executed asynchronously. Some tasks, by nature, takes a lot of time to complete. Downloading a file, for example. Without concurrency, the processor would have to wait for the task to complete before starting to execute the next task. However, with concurrency we could temporarily suspend the current task, and come back later when the task finishes. **Without using extra computing power.**\*\*\*

### So much for introducing concurrency... now, what is gym ?

OpenAI gym, is a `Python` library that helps research reinforcement learning. Reinforcement learning is a branch from control theory, and focusing mainly on agents interacting with environments. And OpenAI gym provides numerous environments for people to benchmark their beloved reinforcement learning algorithms. For you agents to _train_ in a _gym_, they say.

### Um, so why do we need agymc, do you say ?

Despite its merits, OpenAI gym has one major drawback. It is designed to run _one agent on a processor at a time, only_. What if you want to run multiple environments on the same processor at a time? Well, it will run, **sequentially**. Which means slow if you want to train a robot in _batches_.

### Experiments

Using `env.render` as our bottlenecking operation, runing 200 environments, our version`agymc` completes 50 episodes in 4 minutes, while naive `gym` version takes around twice as long. This is what the madness looks like:

![Screenshot_1](./assets/Screenshot_1.png)

### Wow, how to use agymc ?

`agymc`, which combines the power of `Python` async API and OpenAI gym, hence the name, designed for users to make minimal changes to their OpenAI gym code. All usages are the same, except now the returns are in _batches_ (lists), and except serveral environments are now run concurrently. Example below!

### Sounds nice. How to I get it ?

 ![mit](https://img.shields.io/badge/license-MIT-green.svg) ![os](https://img.shields.io/badge/platform-linux%20%7C%20osx-blue.svg) ![py](https://img.shields.io/badge/python-%3E=_3.7-red.svg)

```shell
pip3 install agymc
```

And that's it! Hooray!

### Example Usage Code Snippet

```python
import argparse
import asyncio

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
            # using asyncio.sleep to simulate workflow
            # asyncio.sleep blocks the current thread
            # however we wrapped the environment in a rather nice way
            # such that concurrency still applies
            # the result: It won't block.
            # also worth noting that using this "blocking call"
            # runs faster than having this function do nothing
            # I guess its because the the asyncio.sleep method
            # forces the event loop to schedule thing more nicely
            def function(number):
                asyncio.create_task(asyncio.sleep(1))

            _ = envs.parallel(function, [num_envs * [1]])
            (_, _, done, _) = envs.step(action)
    envs.close()
```



\* When doing pure `gym` operation such as sampling, stepping, this library runs slower since this is a wrapper for gym. However, for actions that _takes a while to execute, such as backprop and update, sending data back and forth, or even rendering_, concurrency makes the operations execute much faster than a [naive gym implementation](./test/ref.py)

\*\* If you would like to learn more about concurrency patterns, [this](https://www.youtube.com/watch?v=rDRa23k70CU) video is really informative.

\*\*\* Without using extra computing power, save a tiny overhead called scheduler. Which every computer provides and is hard to profile its efficiency.