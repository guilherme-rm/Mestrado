"""Neural network architectures for agents."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class DNN(nn.Module):
    """Simple feedforward DQN backbone.
    Input: Global state vector (size: nagents).
    Output: Q-values for all actions (size: BS_Number * nChannel).
    """

    def __init__(self, opt, sce, scenario):
        super().__init__()
        input_dim = opt.nagents
        output_dim = scenario.BS_Number() * sce.nChannel

        # Increased network capacity for better representation learning
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_dim)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # Xavier initialization for stability
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Ensure input tensor is float32 for compatibility with linear layers
        x = state.to(torch.float32)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.output_layer(x)


__all__ = ["DNN"]
