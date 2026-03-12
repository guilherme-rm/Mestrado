"""GNN modules using PyTorch Geometric.

Uses PyG's optimized implementations for graph operations:
- GCNConv, GATConv, SAGEConv for message passing
- Data class for efficient batching
- Built-in sparse operations and CUDA support

Components:
- WirelessGraphBuilder: Constructs PyG graphs from network state
- GNNEncoder: Standard GNN with GCN/GAT/SAGE layers
- EdgeConditionedGNN: Uses edge features via NNConv
- GNNObservationEncoder: Integration with MARL pipeline
"""

from .graph_builder import WirelessGraphBuilder, WirelessGraph
from .gnn_encoder import GNNEncoder, EdgeConditionedGNN
from .observation_encoder import GNNObservationEncoder

__all__ = [
    "WirelessGraphBuilder",
    "WirelessGraph",
    "GNNEncoder",
    "EdgeConditionedGNN",
    "GNNObservationEncoder",
]
