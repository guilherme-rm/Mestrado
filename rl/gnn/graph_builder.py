"""Wireless network graph construction using PyTorch Geometric.

Uses PyG's Data class for efficient batching and GPU processing.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Union
import numpy as np
import torch
from torch_geometric.data import Data

from telecom.scenario import Scenario


# Node/Edge type encodings
NODE_TYPE_UE, NODE_TYPE_MBS, NODE_TYPE_PBS, NODE_TYPE_FBS, NODE_TYPE_AP = 0, 1, 2, 3, 4
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
        self._bs_positions = np.array([bs.get_location() for bs in self._bs_list], dtype=np.float32)
        self._bs_powers = np.array([bs.Transmit_Power_dBm() for bs in self._bs_list], dtype=np.float32)
        self._bs_radii = np.array([bs.radius for bs in self._bs_list], dtype=np.float32)
        self._bs_types = [bs.bs_type for bs in self._bs_list]
        self._n_channel = scenario.sce.nChannel
        self._n_bs = len(self._bs_list)
        self._type_map = {
            "MBS": NODE_TYPE_MBS,
            "PBS": NODE_TYPE_PBS,
            "FBS": NODE_TYPE_FBS,
            "AP": NODE_TYPE_AP,
        }
        # Max type code used for [0,1] normalization; update if new types are added
        self._type_norm = float(max(self._type_map.values()))
        self._bs_positions_t = torch.as_tensor(self._bs_positions, device=self.device, dtype=torch.float32)
        self._bs_powers_t = torch.as_tensor(self._bs_powers, device=self.device, dtype=torch.float32)
        self._bs_radii_t = torch.as_tensor(self._bs_radii, device=self.device, dtype=torch.float32)

    def _to_pos_tensor(self, positions: List[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
        if not positions:
            return torch.zeros((0, 2), device=self.device, dtype=torch.float32)
        rows = [torch.as_tensor(p, device=self.device, dtype=torch.float32) for p in positions]
        return torch.stack(rows, dim=0)
    
    def build(self, ue_positions: List[Union[np.ndarray, torch.Tensor]], actions: Optional[List[int]] = None,
              ue_sinrs: Optional[List[float]] = None) -> WirelessGraph:
        ue_pos_t = self._to_pos_tensor(ue_positions)
        n_ues = int(ue_pos_t.size(0))
        
        # Decode actions
        bs_assign = [int(a) // self._n_channel if a is not None else -1 for a in (actions or [-1]*n_ues)]
        channels = [int(a) % self._n_channel if a is not None else -1 for a in (actions or [-1]*n_ues)]
        bs_loads = [sum(1 for b in bs_assign if b == i) for i in range(self._n_bs)]
        
        # Build features and edges
        x = self._build_features(ue_pos_t, bs_assign, channels, ue_sinrs, bs_loads)
        edge_index, edge_attr = self._build_edges(ue_pos_t, bs_assign, channels, n_ues)
        
        return WirelessGraph(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            num_ues=n_ues, num_bs=self._n_bs).to(self.device)
    
    def _build_features(self, ue_pos: torch.Tensor, bs_assign, channels, sinrs, bs_loads):
        n_ues, n_bs, feat_dim = len(ue_pos), self._n_bs, 8
        max_ue_pos = float(ue_pos.abs().max()) if n_ues > 0 else 0.0
        max_bs_pos = float(self._bs_positions_t.abs().max()) if self._n_bs > 0 else 0.0
        max_pos = max(max_ue_pos, max_bs_pos) + 1e-6
        max_r = float(self._bs_radii_t.max()) if self._n_bs > 0 else 500.0
        
        x = torch.zeros(n_ues + n_bs, feat_dim, device=self.device, dtype=torch.float32)
        
        for i, pos in enumerate(ue_pos):
            bs = bs_assign[i]
            x[i, :2] = pos / max_pos
            x[i, 2] = (sinrs[i] + 10) / 50.0 if sinrs and i < len(sinrs) else 0.5
            x[i, 3] = (channels[i] + 1) / (self._n_channel + 1) if channels[i] >= 0 else 0
            x[i, 4] = (bs + 1) / (n_bs + 1) if bs >= 0 else 0
            if 0 <= bs < n_bs:
                d = torch.linalg.vector_norm(pos - self._bs_positions_t[bs]).clamp_min(1.0)
                x[i, 5] = torch.minimum(d / max_r, torch.tensor(1.0, device=self.device))
                x[i, 6] = (self._bs_powers_t[bs] - 20.0 * torch.log10(d) + 120.0) / 150.0
        
        for j in range(n_bs):
            x[n_ues+j, :2] = self._bs_positions_t[j] / max_pos
            x[n_ues+j, 2] = self._bs_powers_t[j] / 50.0
            x[n_ues+j, 3] = self._bs_radii_t[j] / max_r
            x[n_ues+j, 4] = self._type_map.get(self._bs_types[j], NODE_TYPE_MBS) / self._type_norm
            x[n_ues+j, 5] = bs_loads[j] / max(len(ue_pos), 1)
        
        return x
    
    def _build_edges(self, ue_pos: torch.Tensor, bs_assign, channels, n_ues):
        src, dst, feats = [], [], []
        max_r = float(self._bs_radii_t.max()) if self._n_bs > 0 else 500.0
        
        def add_edge(s, d, dist, etype):
            src.extend([s, d]); dst.extend([d, s])
            dist_t = torch.as_tensor(dist, device=self.device, dtype=torch.float32).clamp_min(1.0)
            frac = torch.minimum(dist_t / max_r, torch.tensor(1.0, device=self.device))
            f = torch.stack(
                [
                    frac,
                    20.0 * torch.log10(dist_t) / 100.0,
                    torch.tensor(float(etype) / 2.0, device=self.device),
                    1.0 - frac,
                ]
            )
            feats.extend([f, f])
        
        # Communication edges
        for i, bs in enumerate(bs_assign):
            if 0 <= bs < self._n_bs:
                add_edge(i, n_ues+bs, torch.linalg.vector_norm(ue_pos[i] - self._bs_positions_t[bs]), EDGE_TYPE_COMM)
        
        # Potential edges
        if self.include_potential_links:
            for i, pos in enumerate(ue_pos):
                for bs in range(self._n_bs):
                    if bs != bs_assign[i]:
                        d = torch.linalg.vector_norm(pos - self._bs_positions_t[bs])
                        if float(d) <= min(float(self._bs_radii_t[bs]), PROXIMITY_THRESHOLD_METERS):
                            add_edge(i, n_ues+bs, d, EDGE_TYPE_POTENTIAL)
        
        # Interference edges
        if self.include_interference_edges:
            ch_ues: Dict[int, List[int]] = {}
            for i, ch in enumerate(channels):
                if ch >= 0: ch_ues.setdefault(ch, []).append(i)
            for ues in ch_ues.values():
                for i in range(len(ues)):
                    for j in range(i+1, len(ues)):
                        add_edge(ues[i], ues[j], torch.linalg.vector_norm(ue_pos[ues[i]] - ue_pos[ues[j]]), EDGE_TYPE_INTERF)
        
        if src:
            return (
                torch.tensor([src, dst], dtype=torch.long, device=self.device),
                torch.stack(feats).to(device=self.device, dtype=torch.float32),
            )
        return (
            torch.zeros(2, 0, dtype=torch.long, device=self.device),
            torch.zeros(0, 4, dtype=torch.float32, device=self.device),
        )


__all__ = ["WirelessGraphBuilder", "WirelessGraph"]
