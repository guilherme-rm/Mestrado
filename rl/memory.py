"""Replay memory utilities."""

from __future__ import annotations

# Use deque for efficient memory management
from collections import namedtuple, deque
from random import sample

import torch

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


class TensorReplayMemory:
    """Preallocated tensor-backed replay buffer.

    Lazily initialises storage on the first Push call so state dimensions
    do not need to be known at construction time.  Sample() returns
    pre-stacked tensors directly, eliminating the zip/cat overhead of
    the legacy deque-based buffer.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._size = 0
        self._pos = 0
        # Lazily initialised on first Push
        self._states: torch.Tensor = None
        self._actions: torch.Tensor = None
        self._next_states: torch.Tensor = None
        self._rewards: torch.Tensor = None
        self._device = None

    def _init_buffers(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor):
        state_dim = state.shape[-1]
        action_dim = action.shape[-1]
        self._device = state.device
        self._states = torch.zeros(self.capacity, state_dim, dtype=state.dtype, device=self._device)
        self._actions = torch.zeros(self.capacity, action_dim, dtype=torch.long, device=self._device)
        self._next_states = torch.zeros(self.capacity, state_dim, dtype=state.dtype, device=self._device)
        self._rewards = torch.zeros(self.capacity, dtype=reward.dtype, device=self._device)

    def Push(self, state: torch.Tensor, action: torch.Tensor,
             next_state: torch.Tensor, reward: torch.Tensor):
        """Store a transition (state shape (1,D), action (1,1), next_state (1,D), reward (1,))."""
        if self._states is None:
            self._init_buffers(state, action, reward)
        pos = self._pos
        self._states[pos] = state.squeeze(0)
        self._actions[pos] = action.squeeze(0)
        self._next_states[pos] = next_state.squeeze(0)
        self._rewards[pos] = reward.squeeze()
        self._pos = (pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def Sample(self, batch_size: int):
        """Return (states, actions, next_states, rewards) as pre-stacked tensors."""
        indices = torch.randint(0, self._size, (batch_size,), device=self._device)
        return (
            self._states[indices],
            self._actions[indices],
            self._next_states[indices],
            self._rewards[indices],
        )

    def __len__(self):
        return self._size


__all__ = ["Transition", "ReplayMemory", "TensorReplayMemory"]
