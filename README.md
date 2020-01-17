# agymc

An asynchronous wrapper for OpenAI Gym library.

Gym, despite its merits, runs pretty slow and can only interact with one agent at one time.

What if we want to run several agents at the same time?

Agymc answers the question by wrapping gym library, while attempting to schedule several gym environments to run at the same time, courtesy of python's `asyncio` library.

