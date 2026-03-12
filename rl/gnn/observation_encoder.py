"""GNN-based observation encoder for MARL agents.

Integrates the GNN encoder with the MARL observation pipeline.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Literal
import torch
import torch.nn as nn
import numpy as np

from telecom.scenario import Scenario
from constants import (
    GNN_HIDDEN_DIM, GNN_OUTPUT_DIM, GNN_NUM_LAYERS,
    GNN_USE_ATTENTION, GNN_INCLUDE_INTERFERENCE_EDGES,
)
from rl.gnn.graph_builder import WirelessGraphBuilder, WirelessGraph
from rl.gnn.gnn_encoder import GNNEncoder, EdgeConditionedGNN


class GNNObservationEncoder(nn.Module):
    """Encodes network state into per-agent GNN embeddings.
    
    Pipeline:
    1. Build graph from current network state
    2. Run GNN to compute node embeddings  
    3. Extract UE embeddings as agent observations
    
    Args:
        scenario: Telecom scenario
        device: PyTorch device
        mode: "replace" (GNN only) or "augment" (GNN + flat obs)
        gnn_output_dim: Output embedding dimension
        conv_type: GNN architecture ("gcn", "gat", "sage", "edge")
    """
    
    def __init__(
        self,
        scenario: Scenario,
        device: torch.device,
        mode: Literal["replace", "augment"] = "replace",
        gnn_output_dim: int = GNN_OUTPUT_DIM,
        gnn_hidden_dim: int = GNN_HIDDEN_DIM,
        gnn_num_layers: int = GNN_NUM_LAYERS,
        use_attention: bool = GNN_USE_ATTENTION,
        include_interference_edges: bool = GNN_INCLUDE_INTERFERENCE_EDGES,
        conv_type: str = "gcn",
        flat_obs_dim: int = 0,
        heterogeneous: bool = False,  # Kept for API compatibility
    ):
        super().__init__()
        
        self.device = device
        self.mode = mode
        self.gnn_output_dim = gnn_output_dim
        
        # Graph builder
        self.graph_builder = WirelessGraphBuilder(
            scenario=scenario,
            device=device,
            include_interference_edges=include_interference_edges,
        )
        
        # GNN encoder - choose architecture
        if conv_type == "edge":
            self.gnn = EdgeConditionedGNN(
                input_dim=8,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_output_dim,
                num_layers=gnn_num_layers,
            )
        else:
            gnn_type = "gat" if use_attention else conv_type
            self.gnn = GNNEncoder(
                input_dim=8,
                hidden_dim=gnn_hidden_dim,
                output_dim=gnn_output_dim,
                num_layers=gnn_num_layers,
                conv_type=gnn_type,
            )
        
        self._output_dim = gnn_output_dim
    
    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def forward(
        self,
        ue_positions: List[np.ndarray],
        actions: Optional[List[int]] = None,
        ue_sinrs: Optional[List[float]] = None,
        flat_observations: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Compute GNN observations for all agents."""
        # Build graph
        graph = self.graph_builder.build(ue_positions, actions, ue_sinrs)
        
        # Get embeddings
        node_emb = self.gnn(graph)
        ue_emb = graph.get_ue_embeddings(node_emb)
        
        return {i: ue_emb[i] for i in range(len(ue_positions))}


__all__ = ["GNNObservationEncoder"]
