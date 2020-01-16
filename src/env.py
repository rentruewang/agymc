import argparse
import asyncio

import gym
import numpy as np
from tqdm import tqdm


class Container(tuple):
    def __call__(self, *args, **kwargs):
        return Container(it(*args, **kwargs) for it in self)

    def __getattr__(self, name):
        return Container(it.__getattribute__(name) for it in self)


class Env(Container):
    def __call__(self, *args, **kwargs):
        raise AttributeError

    def reset(self):
        return Container(
            asyncio.get_event_loop().run_until_complete(
                asyncio.gather(*(self._reset(env) for env in self))
            )
        )

    async def _reset(self, env):
        env.done = False
        return env.reset()

    def render(self):
        return Container(
            asyncio.get_event_loop().run_until_complete(
                asyncio.gather(*(self._render(env) for env in self))
            )
        )

    async def _render(self, env):
        return env.render()

    def step(self, actions):
        return zip(
            *asyncio.get_event_loop().run_until_complete(
                asyncio.gather(
                    *(self._step(env, action) for (env, action) in zip(self, actions))
                )
            )
        )

    async def _step(self, env, action):
        if not env.done:
            out = env.step(action)
            if out[2]:
                env.done = True
            return out
        else:
            return (None, None, True, None)


def make(name_env, num_envs):
    return Env((gym.make(name_env) for _ in range(num_envs)))

