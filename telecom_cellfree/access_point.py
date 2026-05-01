"""Access point model for Cell-Free Massive MIMO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set
import numpy as np


@dataclass
class AccessPoint:
    """Distributed AP entity used in user-centric Cell-Free topology."""

    id: int
    position: np.ndarray
    max_tx_power_dbm: float
    radius: float
    antenna_count: int = 1
    cpu_id: int = 0
    bs_type: str = "AP"
    current_load: float = 0.0
    associated_users: Set[int] = field(default_factory=set)

    def reset(self):
        self.current_load = 0.0
        self.associated_users.clear()

    def get_location(self):
        return self.position

    def transmit_power_dbm(self) -> float:
        return self.max_tx_power_dbm

    def receive_power(self, d: float) -> float:
        """Return approximate received power in mW for plotting/compatibility."""
        d = max(float(d), 1.0)
        if d > self.radius:
            return 0.0
        path_loss_db = 30.5 + 36.7 * np.log10(d)
        rx_dbm = self.max_tx_power_dbm - path_loss_db
        return 10 ** (rx_dbm / 10.0)

    # Legacy compatibility names
    def Get_Location(self):  # noqa: N802
        return self.get_location()

    def Transmit_Power_dBm(self):  # noqa: N802
        return self.transmit_power_dbm()

    def Receive_Power(self, d):  # noqa: N802
        return self.receive_power(d)
