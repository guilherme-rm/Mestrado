"""Neural network architectures for agents."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class DNN(nn.Module):
    """Simple feedforward DQN backbone.

    Hidden sizes are fixed for now but can be parameterized.
    """

    def __init__(self, opt, sce, scenario):  # Testar com Kernel Gaussiano -> Camada Linear
        super().__init__()
        self.input_layer = nn.Linear(opt.nagents, 64)
        self.middle1_layer = nn.Linear(64, 32)
        self.middle2_layer = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, scenario.BS_Number() * sce.nChannel)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x1 = F.relu(self.input_layer(state))
        x2 = F.relu(self.middle1_layer(x1))
        x3 = F.relu(self.middle2_layer(x2))
        return self.output_layer(x3)


__all__ = ["DNN"]
