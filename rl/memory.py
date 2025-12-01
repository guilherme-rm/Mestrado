"""Replay memory utilities."""

from __future__ import annotations

# Use deque for efficient memory management
from collections import namedtuple, deque
from random import sample

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        # Initialize deque with maxlen for automatic eviction of old memories
        self.memory = deque([], maxlen=capacity)

    def Push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def Sample(self, batch_size):
        """Samples a batch of transitions."""
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


__all__ = ["Transition", "ReplayMemory"]
