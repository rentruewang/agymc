import argparse
import asyncio

import gym
import numpy as np


class Container(tuple):
    def __call__(self, *args, **kwargs):
        return Container(it(*args, **kwargs) for it in self)

    def __getattr__(self, name):
        return Container(object.__getattribute__(it, name) for it in self)


# ! these functions are methods of Env class
# ! but are defined out here to avoid bounded method overhead
async def _Env_reset(self, i):
    env = self[i]
    env.done = False
    env.reset()


async def _Env_render(self, i):
    self[i].render()


async def _Env_step(self, i, action):
    env = self[i]
    if not env.done:
        out = env.step(action)
        if out[2]:
            env.done = True
        return out
    else:
        return (None, None, True, None)


async def _Env_close(self, i):
    self[i].close()


async def _Env_seed(self, i, seed):
    self[i].seed(seed)


class Env(Container):
    def __call__(self, *args, **kwargs):
        raise AttributeError

    def reset(self):
        asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*(_Env_reset(self, i) for i in range(len(self))))
        )

    def render(self):
        asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*(_Env_render(self, i) for i in range(len(self))))
        )

    def step(self, actions):
        return zip(
            *asyncio.get_event_loop().run_until_complete(
                asyncio.gather(
                    *(
                        _Env_step(self, i, action)
                        for (i, action) in zip(range(len(self)), actions)
                    )
                )
            )
        )

    def close(self):
        return asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*(_Env_close(self, i) for i in range(len(self))))
        )

    def seed(self, seeds):
        if hasattr(seeds, "__iter__"):
            return asyncio.get_event_loop().run_until_complete(
                asyncio.gather(
                    *(_Env_seed(self, i, s) for (i, s) in zip(range(len(self)), seeds))
                )
            )
        else:
            return asyncio.get_event_loop().run_until_complete(
                asyncio.gather(*(_Env_seed(self, i, seeds) for i in range(len(self))))
            )


def make(name_env, num_envs):
    return Env((gym.make(name_env) for _ in range(num_envs)))
