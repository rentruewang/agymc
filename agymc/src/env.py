"""
This where the Env class is defined,
which handles wrapping all enviroments.

FIXME
The asynchronous version speeds up parallel execution by allowing calles that are time consuming to execute in the background.
Since we introduce a lot of python overhead by wrapping calles to iterables with `ObjectHolder` class,
if there are no such blocking calles,
the performance can be as slow as 4 - 5 times slower than the unwrapped version.
Even though the majority of the code is written with ctypes module,
the performance still is not good.
However, this version will not be bottlenecked time consuming operations,
precisely because of the asynchronous nature.

! Yes, this file can be written entirely with ctypes,
! however, that would make its python API much harder to use,
! which defeats its purpose of being syntax sugar,
! for wrapping multiple gym environments concurrently.
"""
import argparse
import asyncio
import ctypes

import gym


class ObjectHolder:
    """
    A holder that calles its member synchronously.
    """

    __slots__ = ["_objects"]

    def __init__(self, objects):
        length = len(objects)
        self._objects = (ctypes.py_object * len(objects))()
        for i in range(length):
            self._objects[i] = objects[i]

    def __getattr__(self, name):
        return ObjectHolder(
            tuple(
                object.__getattribute__(self._objects[i], name)
                for i in range(len(self._objects))
            )
        )

    def __call__(self, *args, **kwargs):
        return ObjectHolder(
            tuple(self._objects[i](*args, **kwargs) for i in range(len(self._objects)))
        )

    def __iter__(self):
        return iter(self._objects)


# ! these functions are methods of Env class
# ! but are defined out here to avoid bounded method overhead
async def _Env_reset(env, *args, **kwargs):
    env.done = False
    return env.reset(*args, **kwargs)


async def _Env_render(env, *args, **kwargs):
    return env.render(*args, **kwargs)


async def _Env_step(env, action, *args, **kwargs):
    if not env.done:
        out = env.step(action, *args, **kwargs)
        if out[2]:
            env.done = True
        return out
    else:
        return (None, None, True, None)


async def _Env_close(env, *args, **kwargs):
    return env.close(*args, **kwargs)


async def _Env_seed(env, seed, *args, **kwargs):
    return env.seed(seed, *args, **kwargs)


async def _Env_compute_reward(env, *args, **kwargs):
    return env.compute_reward(*args, **kwargs)


async def _Env_class_name(env, *args, **kwargs):
    return env.class_name(*args, **kwargs)


async def _Env_parallel(function, *args, **kwargs):
    return function(*args, **kwargs)


def _Env_parallel_arg_param(arguments, args):
    for arg in zip(*arguments):
        yield (*arg, *args)


def _Env_parallel_kwarg_param(kwarguments, kwargs):
    newdict = {k: iter(v) for (k, v) in kwarguments.items()}
    while True:
        try:
            yield dict(**{k: next(v) for (k, v) in newdict.items()}, **kwargs)
        except StopIteration:
            break


class Env:
    """
    A gym Env class wrapper for convenient chaining calles
    e.g. env.action_space.sample() works
    """

    __slots__ = [
        "_iterable",
        "action_space",
        "metadata",
        "observation_space",
        "reward_range",
        "spec",
        "unwrapped",
    ]

    def __init__(self, envs):
        length = len(envs)
        self._iterable = (ctypes.py_object * length)()
        for i in range(length):
            self._iterable[i] = envs[i]

        self.action_space = ObjectHolder(
            tuple(self._iterable[i].action_space for i in range(len(self._iterable)))
        )
        self.metadata = ObjectHolder(
            tuple(self._iterable[i].metadata for i in range(len(self._iterable)))
        )
        self.observation_space = ObjectHolder(
            tuple(
                self._iterable[i].observation_space for i in range(len(self._iterable))
            )
        )
        self.reward_range = ObjectHolder(
            tuple(self._iterable[i].reward_range for i in range(len(self._iterable)))
        )
        self.spec = ObjectHolder(
            tuple(self._iterable[i].spec for i in range(len(self._iterable)))
        )
        self.unwrapped = ObjectHolder(
            tuple(self._iterable[i].unwrapped for i in range(len(self._iterable)))
        )

    def reset(self):
        asyncio.get_event_loop().run_until_complete(
            asyncio.gather(
                *(_Env_reset(self._iterable[i]) for i in range(len(self._iterable)))
            )
        )

    def render(self):
        asyncio.get_event_loop().run_until_complete(
            asyncio.gather(
                *(_Env_render(self._iterable[i]) for i in range(len(self._iterable)))
            )
        )

    def step(self, actions):
        return zip(
            *asyncio.get_event_loop().run_until_complete(
                asyncio.gather(
                    *(
                        _Env_step(self._iterable[i], action)
                        for (i, action) in zip(range(len(self._iterable)), actions)
                    )
                )
            )
        )

    def seed(self, seeds):
        _iterable = sefl._iterable
        if hasattr(seeds, "__iter__"):
            return asyncio.get_event_loop().run_until_complete(
                asyncio.gather(
                    *(
                        _Env_seed(self._iterable[i], s)
                        for (i, s) in zip(range(len(self._iterable)), seeds)
                    )
                )
            )
        else:
            return asyncio.get_event_loop().run_until_complete(
                asyncio.gather(
                    *(
                        _Env_seed(self._iterable[i], seeds)
                        for i in range(len(self._iterable))
                    )
                )
            )

    def close(self):
        return asyncio.get_event_loop().run_until_complete(
            asyncio.gather(
                *(_Env_close(self._iterable[i]) for i in range(len(self._iterable)))
            )
        )

    def compute_reward(self, *args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(
            asyncio.gather(
                *(
                    _Env_compute_reward(self._iterable[i], *args, **kwargs)
                    for i in range(len(self._iterable))
                )
            )
        )

    def class_name(self, *args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(
            asyncio.gather(
                *(
                    _Env_class_name(self._iterable[i], *args, **kwargs)
                    for i in range(len(self._iterable))
                )
            )
        )

    def parallel(self, function, arguments=list(), kwarguments=dict(), *args, **kwargs):
        """
        This method does not exist in the original gym library. For concurrent use.
        `function` is a callable, and the `function` passed in will be executed concurrently.
        arguments, kwarguments contains specific information for each environment.
        This can be used to make batched prediction.

        example:
            >>> def predict(observation):
                ...
                return prediction
            >>> results = env.parallel(predict, argument=[observation_list])
            >>> env.step(results)
            >>> env.parallel(backprop) # usually library do this in parallel automatically
        """
        return asyncio.get_event_loop().run_until_complete(
            asyncio.gather(
                *(
                    _Env_parallel(function, *args, **kwargs)
                    for (args, kwargs) in zip(
                        _Env_parallel_arg_param(arguments, args),
                        _Env_parallel_kwarg_param(kwarguments, kwargs),
                    )
                )
            )
        )


def make(name_env, num_envs):
    """
    make `num_envs` `name_env`s
    e.g. make("CartPole-v0", 100) makes 100 CartPole-v0 environments.
    """
    return Env(tuple(gym.make(name_env) for _ in range(num_envs)))
