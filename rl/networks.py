"""Neural network architectures for agents."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class DNN(nn.Module):
    """Simple feedforward DQN backbone 
    Input: Global state vector (size: nagents).
    Output: Q-values for all actions (size: BS_Number * nChannel).
    """

    def __init__(self, opt, sce, scenario):
        super().__init__()
        input_dim = opt.nagents
        output_dim = scenario.BS_Number() * sce.nChannel

        # Network architecture matching original UARA-DRL: 64 → 32 → 32
        self.input_layer = nn.Linear(input_dim, 64)
        self.middle1_layer = nn.Linear(64, 32)
        self.middle2_layer = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, output_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Ensure input tensor is float32 for compatibility with linear layers
        x = state.to(torch.float32)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.middle1_layer(x))
        x = F.relu(self.middle2_layer(x))
        return self.output_layer(x)


__all__ = ["DNN"]
