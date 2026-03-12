"""Interference detection and graph computation.

This module provides utilities for detecting and visualizing interference
between UEs that are using the same channel.

Interference occurs when two UEs:
1. Select the same frequency channel
2. Are close enough that the interfering signal exceeds a threshold

The threshold is defined relative to the noise floor for physical meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from telecom.scenario import Scenario

INTERFERENCE_THRESHOLD_DB_ABOVE_NOISE = 3.0  # dB above noise floor

SHOW_INTERFERENCE_EDGES = True
INTERFERENCE_EDGE_COLOR = "red"
INTERFERENCE_EDGE_ALPHA_MIN = 0.3
INTERFERENCE_EDGE_ALPHA_MAX = 0.9
INTERFERENCE_EDGE_WIDTH_MIN = 0.5
INTERFERENCE_EDGE_WIDTH_MAX = 2.5

INTERFERENCE_SCALING_MAX_DB = 20.0  # dB above threshold

@dataclass
class InterferenceEdge:
    """Represents interference between two UEs."""
    ue_i: int  # First UE index
    ue_j: int  # Second UE index
    channel: int  # Shared channel
    strength_mw: float  # Interference power in mW (sum of both directions)
    strength_db: float  # Interference power in dB relative to noise
    
    @property
    def normalized_strength(self) -> float:
        """Normalized strength for visualization (0 to 1)."""
        # Clamp to [0, 1] based on dB above threshold
        val = min(1.0, max(0.0, self.strength_db / INTERFERENCE_SCALING_MAX_DB))
        return val

def compute_interference_graph(
    ue_positions: List[np.ndarray],
    actions: List[int],
    scenario: Scenario,
) -> List[InterferenceEdge]:
    """Compute interference edges between UEs.
    
    Two UEs have an interference edge if:
    1. They use the same channel
    2. The mutual interference power exceeds the threshold
    
    Args:
        ue_positions: List of UE positions as numpy arrays [x, y]
        actions: List of action indices (bs_idx * nChannel + ch_idx)
        scenario: Telecom scenario with BS info
        
    Returns:
        List of InterferenceEdge objects (undirected)
    """
    if not SHOW_INTERFERENCE_EDGES:
        return []
    
    edges = []
    bs_list = scenario.Get_BaseStations()
    n_channel = scenario.sce.nChannel
    n_ues = len(actions)
    
    # Compute noise floor (same as in reward calculation)
    noise_mw = 10 ** (scenario.sce.N0 / 10) * scenario.sce.BW
    
    # Threshold in mW (noise * linear factor)
    threshold_mw = noise_mw * (10 ** (INTERFERENCE_THRESHOLD_DB_ABOVE_NOISE / 10))
    
    # Build channel-to-UEs mapping for efficiency
    channel_ues = {}
    for ue_idx, action in enumerate(actions):
        if action is None:
            continue
        ch = int(action) % n_channel
        if ch not in channel_ues:
            channel_ues[ch] = []
        channel_ues[ch].append(ue_idx)
    
    # Check pairs on same channel
    for ch, ue_indices in channel_ues.items():
        if len(ue_indices) < 2:
            continue
        
        for i in range(len(ue_indices)):
            for j in range(i + 1, len(ue_indices)):
                ue_i = ue_indices[i]
                ue_j = ue_indices[j]
                
                # Get BS selections
                bs_i = int(actions[ue_i]) // n_channel
                bs_j = int(actions[ue_j]) // n_channel
                
                if bs_i >= len(bs_list) or bs_j >= len(bs_list):
                    continue
                
                pos_i = ue_positions[ue_i]
                pos_j = ue_positions[ue_j]
                
                # Interference at UE_i from BS serving UE_j
                bs_j_loc = np.array(bs_list[bs_j].get_location())
                dist_i_to_bsj = np.linalg.norm(pos_i - bs_j_loc)
                interf_at_i = bs_list[bs_j].receive_power(max(1.0, dist_i_to_bsj))
                
                # Interference at UE_j from BS serving UE_i
                bs_i_loc = np.array(bs_list[bs_i].get_location())
                dist_j_to_bsi = np.linalg.norm(pos_j - bs_i_loc)
                interf_at_j = bs_list[bs_i].receive_power(max(1.0, dist_j_to_bsi))
                
                # Total interference (undirected: sum both)
                total_interf = interf_at_i + interf_at_j
                
                if total_interf > threshold_mw:
                    # Convert to dB above noise for visualization
                    strength_db = 10 * np.log10(total_interf / noise_mw)
                    
                    edges.append(InterferenceEdge(
                        ue_i=ue_i,
                        ue_j=ue_j,
                        channel=ch,
                        strength_mw=total_interf,
                        strength_db=strength_db,
                    ))
    
    return edges


def get_interference_stats(edges: List[InterferenceEdge]) -> dict:
    """Compute statistics about interference.
    
    Args:
        edges: List of interference edges
        
    Returns:
        Dictionary with interference statistics
    """
    if not edges:
        return {
            "num_edges": 0,
            "max_strength_db": 0.0,
            "mean_strength_db": 0.0,
            "affected_ues": 0,
        }
    
    strengths = [e.strength_db for e in edges]
    affected = set()
    for e in edges:
        affected.add(e.ue_i)
        affected.add(e.ue_j)
    
    return {
        "num_edges": len(edges),
        "max_strength_db": max(strengths),
        "mean_strength_db": sum(strengths) / len(strengths),
        "affected_ues": len(affected),
    }


__all__ = [
    "compute_interference_graph",
    "get_interference_stats",
    "InterferenceEdge",
    "SHOW_INTERFERENCE_EDGES",
    "INTERFERENCE_EDGE_COLOR",
]
