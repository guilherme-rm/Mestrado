"""Replay memory utilities."""
from __future__ import annotations

from collections import namedtuple
from random import sample

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def Push(self, *args):  # noqa: N802 (legacy method name)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def Sample(self, batch_size):  # noqa: N802
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


__all__ = ["Transition", "ReplayMemory"]
