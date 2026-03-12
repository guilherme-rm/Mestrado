"""GNN encoder using PyTorch Geometric layers.

Uses standard PyG layers (GCNConv, GATConv, SAGEConv) instead of manual
message passing implementation.
"""

from __future__ import annotations

from typing import Literal
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphNorm

from rl.gnn.graph_builder import WirelessGraph


# Default configuration
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 2


class GNNEncoder(nn.Module):
    """GNN encoder using PyTorch Geometric layers.
    
    Supports multiple GNN architectures:
    - "gcn": Graph Convolutional Network (Kipf & Welling, 2017)
    - "gat": Graph Attention Network (Veličković et al., 2018)
    - "sage": GraphSAGE (Hamilton et al., 2017)
    
    Args:
        input_dim: Input node feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        num_layers: Number of GNN layers
        conv_type: Type of convolution ("gcn", "gat", "sage")
        dropout: Dropout probability
        use_edge_attr: Whether to use edge features (GAT only)
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int = DEFAULT_HIDDEN_DIM,
        num_layers: int = DEFAULT_NUM_LAYERS,
        conv_type: Literal["gcn", "gat", "sage"] = "gcn",
        dropout: float = 0.1,
        use_edge_attr: bool = False,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            if conv_type == "gcn":
                self.convs.append(GCNConv(in_dim, out_dim))
            elif conv_type == "gat":
                # GAT with 4 heads, concatenated in hidden layers
                heads = 4 if i < num_layers - 1 else 1
                self.convs.append(GATConv(in_dim, out_dim // heads if i < num_layers - 1 else out_dim, 
                                         heads=heads, dropout=dropout))
            elif conv_type == "sage":
                self.convs.append(SAGEConv(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.norms.append(GraphNorm(out_dim if conv_type != "gat" or i == num_layers - 1 
                                        else out_dim))
        
        self.activation = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, graph: WirelessGraph) -> torch.Tensor:
        """Forward pass through GNN.
        
        Args:
            graph: WirelessGraph (PyG Data object)
            
        Returns:
            Node embeddings of shape (num_nodes, output_dim)
        """
        x = self.input_proj(graph.x)
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, graph.edge_index)
            x = norm(x)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = self.drop(x)
        
        return x
    
    def get_ue_embeddings(self, graph: WirelessGraph) -> torch.Tensor:
        """Get embeddings for UE nodes only."""
        all_emb = self.forward(graph)
        return graph.get_ue_embeddings(all_emb)


class EdgeConditionedGNN(nn.Module):
    """GNN that uses edge features via NNConv (edge-conditioned convolution).
    
    Better for wireless networks where edge features (distance, path loss)
    are important.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        output_dim: int = DEFAULT_HIDDEN_DIM,
        edge_dim: int = 4,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = 0.1,
    ):
        super().__init__()
        from torch_geometric.nn import NNConv
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            # Edge network: maps edge features to weight matrix
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, 32),
                nn.ReLU(),
                nn.Linear(32, in_dim * out_dim),
            )
            self.convs.append(NNConv(in_dim, out_dim, edge_nn, aggr='mean'))
            self.norms.append(GraphNorm(out_dim))
        
        self.activation = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        self.num_layers = num_layers
    
    def forward(self, graph: WirelessGraph) -> torch.Tensor:
        x = self.input_proj(graph.x)
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, graph.edge_index, graph.edge_attr)
            x = norm(x)
            if i < self.num_layers - 1:
                x = self.activation(x)
                x = self.drop(x)
        
        return x


__all__ = ["GNNEncoder", "EdgeConditionedGNN"]
