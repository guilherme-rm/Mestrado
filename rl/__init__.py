"""Reinforcement learning package.

Includes neural network architectures, replay memory, agent logic,
and GNN-based feature extraction (using PyTorch Geometric).

Public API shortcuts:
    from rl.networks import DNN
    from rl.memory import Transition, ReplayMemory
    from rl.agent import Agent
    from rl.gnn import GNNObservationEncoder, WirelessGraphBuilder
"""

from .memory import Transition, ReplayMemory  # noqa: F401
from .networks import DNN  # noqa: F401
from .agent import Agent  # noqa: F401

# GNN exports (using PyTorch Geometric)
from .gnn import (  # noqa: F401
    GNNObservationEncoder,
    WirelessGraphBuilder,
    WirelessGraph,
    GNNEncoder,
    EdgeConditionedGNN,
)

__all__ = [
    "Transition",
    "ReplayMemory",
    "DNN",
    "Agent",
    # GNN components
    "GNNObservationEncoder",
    "WirelessGraphBuilder",
    "WirelessGraph",
    "GNNEncoder",
    "EdgeConditionedGNN",
]
