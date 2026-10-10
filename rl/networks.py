"""Neural network architectures for agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]

    def to_dict(self):
        return asdict(self)


def resolve_model_spec(opt, sce, scenario, input_dim=None):
    get_option = opt.get if isinstance(opt, dict) else lambda name, default=None: getattr(opt, name, default)
    architecture = get_option('architecture', 'mlp')
    if architecture != 'mlp':
        raise ValueError(f'Unsupported action-value architecture: {architecture}')
    resolved_input_dim = int(input_dim if input_dim is not None else get_option('nagents'))
    output_dim = int(scenario.BS_Number() * sce.nChannel)
    hidden_dims = tuple(int(size) for size in (get_option('dnn_hidden_layers') or [128, 64, 64]))
    if resolved_input_dim < 1 or output_dim < 1 or any(size < 1 for size in hidden_dims):
        raise ValueError('Model dimensions must be positive')
    return ModelSpec(architecture, resolved_input_dim, output_dim, hidden_dims)


class DNN(nn.Module):
    """Simple feedforward DQN backbone 
    Input: Global state vector (size: nagents) or GNN embedding (size: input_dim).
    Output: Q-values for all actions (size: BS_Number * nChannel).
    
    When GNN is enabled, pass input_dim explicitly. Otherwise uses nagents.
    """

    def __init__(self, opt, sce, scenario, input_dim: int = None, model_spec: ModelSpec = None):
        super().__init__()
        model_spec = model_spec or resolve_model_spec(opt, sce, scenario, input_dim)
        self._input_dim = model_spec.input_dim
        dims = [model_spec.input_dim, *model_spec.hidden_dims, model_spec.output_dim]
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


def build_model(opt, sce, scenario, input_dim=None):
    spec = resolve_model_spec(opt, sce, scenario, input_dim)
    return DNN(opt, sce, scenario, model_spec=spec)
