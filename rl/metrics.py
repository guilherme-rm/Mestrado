from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import torch.nn as nn

@dataclass
class NetworkMetrics:
    """Stores metrics from a single optimization step."""
    loss: float = 0.0
    mean_q: float = 0.0
    max_q: float = 0.0
    min_q: float = 0.0
    q_std: float = 0.0
    grad_norm: float = 0.0
    grad_clipped: bool = False
    target_q_mean: float = 0.0
    td_error: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss": self.loss,
            "mean_q": self.mean_q,
            "max_q": self.max_q,
            "min_q": self.min_q,
            "q_std": self.q_std,
            "grad_norm": self.grad_norm,
            "grad_clipped": float(self.grad_clipped),
            "target_q_mean": self.target_q_mean,
            "td_error": self.td_error
        }
    
class NetworkMetricsTracker:
    """Aggregates network metrics over training."""

    def __init__(self):
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "mean_q": [],
            "max_q": [],
            "min_q": [],
            "q_std": [],
            "grad_norm":[],
            "grad_clipped": [],
            "target_q_mean": [],
            "td_error": []
        }

    def record(self, metrics: NetworkMetrics):
        for key, value in metrics:
            self.history[key].append(value)

    def get_recent_mean(self, key: str, window: int = 100) -> float:

        if key not in self.history or not self.history[key]:
            return 0.0
        data = self.history[key][-window:]

        return sum(data) / len(data)
    
def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5

__all__ = ["NetworkMetrics", "NetworkMetricsTracker", "compute_gradient_norm"]