"""Wireless network graph construction using PyTorch Geometric.

Uses PyG's Data class for efficient batching and GPU processing.
"""

from __future__ import annotations

from typing import List, Optional, Dict
import numpy as np
import torch
from torch_geometric.data import Data

from telecom.scenario import Scenario


# Node/Edge type encodings
NODE_TYPE_UE, NODE_TYPE_MBS, NODE_TYPE_PBS, NODE_TYPE_FBS = 0, 1, 2, 3
EDGE_TYPE_COMM, EDGE_TYPE_POTENTIAL, EDGE_TYPE_INTERF = 0, 1, 2
PROXIMITY_THRESHOLD_METERS = 200.0


class WirelessGraph(Data):
    """PyG Data subclass for wireless network graphs."""
    
    def __init__(self, x=None, edge_index=None, edge_attr=None, num_ues=0, num_bs=0, **kwargs):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, **kwargs)
        self.num_ues = num_ues
        self.num_bs = num_bs
    
    def get_ue_embeddings(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        return node_embeddings[:self.num_ues]


class WirelessGraphBuilder:
    """Constructs PyG graphs from wireless network state."""
    
    def __init__(self, scenario: Scenario, device: torch.device,
                 include_potential_links: bool = True,
                 include_interference_edges: bool = True):
        self.scenario = scenario
        self.device = device
        self.include_potential_links = include_potential_links
        self.include_interference_edges = include_interference_edges
        
        # Cache BS info
        self._bs_list = scenario.Get_BaseStations()
        self._bs_positions = np.array([bs.get_location() for bs in self._bs_list])
        self._bs_powers = np.array([bs.Transmit_Power_dBm() for bs in self._bs_list])
        self._bs_radii = np.array([bs.radius for bs in self._bs_list])
        self._bs_types = [bs.bs_type for bs in self._bs_list]
        self._n_channel = scenario.sce.nChannel
        self._n_bs = len(self._bs_list)
        self._type_map = {"MBS": NODE_TYPE_MBS, "PBS": NODE_TYPE_PBS, "FBS": NODE_TYPE_FBS}
    
    def build(self, ue_positions: List[np.ndarray], actions: Optional[List[int]] = None,
              ue_sinrs: Optional[List[float]] = None) -> WirelessGraph:
        n_ues = len(ue_positions)
        
        # Decode actions
        bs_assign = [int(a) // self._n_channel if a is not None else -1 for a in (actions or [-1]*n_ues)]
        channels = [int(a) % self._n_channel if a is not None else -1 for a in (actions or [-1]*n_ues)]
        bs_loads = [sum(1 for b in bs_assign if b == i) for i in range(self._n_bs)]
        
        # Build features and edges
        x = self._build_features(ue_positions, bs_assign, channels, ue_sinrs, bs_loads)
        edge_index, edge_attr = self._build_edges(ue_positions, bs_assign, channels, n_ues)
        
        return WirelessGraph(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            num_ues=n_ues, num_bs=self._n_bs).to(self.device)
    
    def _build_features(self, ue_pos, bs_assign, channels, sinrs, bs_loads):
        n_ues, n_bs, feat_dim = len(ue_pos), self._n_bs, 8
        max_pos = max(max(abs(p).max() for p in ue_pos), abs(self._bs_positions).max()) + 1e-6
        max_r = max(self._bs_radii) if len(self._bs_radii) > 0 else 500.0
        
        x = torch.zeros(n_ues + n_bs, feat_dim)
        
        for i, pos in enumerate(ue_pos):
            bs = bs_assign[i]
            x[i, :2] = torch.tensor(pos) / max_pos
            x[i, 2] = (sinrs[i] + 10) / 50.0 if sinrs and i < len(sinrs) else 0.5
            x[i, 3] = (channels[i] + 1) / (self._n_channel + 1) if channels[i] >= 0 else 0
            x[i, 4] = (bs + 1) / (n_bs + 1) if bs >= 0 else 0
            if 0 <= bs < n_bs:
                d = np.linalg.norm(pos - self._bs_positions[bs])
                x[i, 5] = min(d / max_r, 1.0)
                x[i, 6] = (self._bs_powers[bs] - 20*np.log10(max(d,1)) + 120) / 150
        
        for j in range(n_bs):
            x[n_ues+j, :2] = torch.tensor(self._bs_positions[j]) / max_pos
            x[n_ues+j, 2] = self._bs_powers[j] / 50.0
            x[n_ues+j, 3] = self._bs_radii[j] / max_r
            x[n_ues+j, 4] = self._type_map.get(self._bs_types[j], 1) / 3.0
            x[n_ues+j, 5] = bs_loads[j] / max(len(ue_pos), 1)
        
        return x
    
    def _build_edges(self, ue_pos, bs_assign, channels, n_ues):
        src, dst, feats = [], [], []
        max_r = max(self._bs_radii) if len(self._bs_radii) > 0 else 500.0
        
        def add_edge(s, d, dist, etype):
            src.extend([s, d]); dst.extend([d, s])
            f = torch.tensor([min(dist/max_r, 1), 20*np.log10(max(dist,1))/100, etype/2, 1-min(dist/max_r,1)])
            feats.extend([f, f])
        
        # Communication edges
        for i, bs in enumerate(bs_assign):
            if 0 <= bs < self._n_bs:
                add_edge(i, n_ues+bs, np.linalg.norm(ue_pos[i]-self._bs_positions[bs]), EDGE_TYPE_COMM)
        
        # Potential edges
        if self.include_potential_links:
            for i, pos in enumerate(ue_pos):
                for bs in range(self._n_bs):
                    if bs != bs_assign[i]:
                        d = np.linalg.norm(pos - self._bs_positions[bs])
                        if d <= min(self._bs_radii[bs], PROXIMITY_THRESHOLD_METERS):
                            add_edge(i, n_ues+bs, d, EDGE_TYPE_POTENTIAL)
        
        # Interference edges
        if self.include_interference_edges:
            ch_ues: Dict[int, List[int]] = {}
            for i, ch in enumerate(channels):
                if ch >= 0: ch_ues.setdefault(ch, []).append(i)
            for ues in ch_ues.values():
                for i in range(len(ues)):
                    for j in range(i+1, len(ues)):
                        add_edge(ues[i], ues[j], np.linalg.norm(ue_pos[ues[i]]-ue_pos[ues[j]]), EDGE_TYPE_INTERF)
        
        if src:
            return torch.tensor([src, dst], dtype=torch.long), torch.stack(feats).float()
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 4, dtype=torch.float32)


__all__ = ["WirelessGraphBuilder", "WirelessGraph"]
