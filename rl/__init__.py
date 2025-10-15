"""Reinforcement learning package.

Includes neural network architectures, replay memory, and agent logic.

Public API shortcuts:
    from rl.networks import DNN
    from rl.memory import Transition, ReplayMemory
    from rl.agent import Agent
"""

from .memory import Transition, ReplayMemory  # noqa: F401
from .networks import DNN  # noqa: F401
from .agent import Agent  # noqa: F401

__all__ = ["Transition", "ReplayMemory", "DNN", "Agent"]
