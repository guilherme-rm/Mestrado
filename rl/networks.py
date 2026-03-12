"""Neural network architectures for agents."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class DNN(nn.Module):
    """Simple feedforward DQN backbone 
    Input: Global state vector (size: nagents) or GNN embedding (size: input_dim).
    Output: Q-values for all actions (size: BS_Number * nChannel).
    
    When GNN is enabled, pass input_dim explicitly. Otherwise uses nagents.
    """

    def __init__(self, opt, sce, scenario, input_dim: int = None):
        super().__init__()
        
        # Use explicit input_dim if provided, otherwise default to nagents
        # GNN mode should explicitly pass input_dim=GNN_OUTPUT_DIM
        self._input_dim = input_dim if input_dim is not None else opt.nagents
            
        output_dim = scenario.BS_Number() * sce.nChannel

        # Network architecture matching original UARA-DRL: 128 → 64 → 64
        self.input_layer = nn.Linear(self._input_dim, 128)
        self.middle1_layer = nn.Linear(128, 64)
        self.middle2_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, output_dim)
    
    @property
    def input_dim(self) -> int:
        """Return the input dimension of the network."""
        return self._input_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Ensure input tensor is float32 for compatibility with linear layers
        x = state.to(torch.float32)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.middle1_layer(x))
        x = F.relu(self.middle2_layer(x))
        return self.output_layer(x)


__all__ = ["DNN"]
