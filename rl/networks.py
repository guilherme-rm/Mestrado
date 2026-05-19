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

        # Hidden layer sizes read from opt (set via network config JSON).
        # Falls back to [128, 64, 64] to preserve original UARA-DRL behaviour.
        hidden_layers = list(getattr(opt, "dnn_hidden_layers", None) or [128, 64, 64])
        dims = [self._input_dim] + hidden_layers + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    @property
    def input_dim(self) -> int:
        """Return the input dimension of the network."""
        return self._input_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Ensure input tensor is float32 for compatibility with linear layers
        x = state.to(torch.float32)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


__all__ = ["DNN"]
