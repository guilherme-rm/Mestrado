"""CPU coordination model for Cell-Free Massive MIMO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CPU:
    """Central processing unit coordinating a subset of APs."""

    id: int
    connected_ap_ids: List[int] = field(default_factory=list)
    fronthaul_capacity_mbps: float = 10000.0
    current_load_mbps: float = 0.0

    def reset(self):
        self.current_load_mbps = 0.0

    @property
    def utilization(self) -> float:
        if self.fronthaul_capacity_mbps <= 0:
            return 0.0
        return self.current_load_mbps / self.fronthaul_capacity_mbps
