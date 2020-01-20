# agymc

**For reinforcement learning and concurrency lovers out there ...**

This is a concurrent wrapper for OpenAI Gym library that runs multiple environments concurrently, which means running faster in training\* without consuming more CPU power.

### What exactly is _concurrency_ ?

Maybe you have heard of _parallel computing_ ? When we say we execute things in parallel, we run the program on _multiple_ processers, which offers significant speedup. _Concurrency computing_ has a broader meaning, though. The definition of a _concurrent_ program, is that it is designed to not execute sequentially.

\* When doing pure `gym` operation such as sampling, stepping, this library runs slower since this is a wrapper for gym. However, for actions that _takes a while to execute, such as backprop and update, sending data back and forth, or even rendering_, concurrency makes the operations execute much faster than a [naive gym implementation](./test/ref.py)