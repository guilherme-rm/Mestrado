"""Base station entity and radio related utilities."""
from __future__ import annotations

import numpy as np

TX_POWER_DBM = {"MBS": 40, "PBS": 30, "FBS": 20}


class BS:
    """Represents a base station with type-specific attributes.

    Provides both snake_case and legacy CamelCase accessors for backward
    compatibility with earlier code.
    """

    def __init__(self, sce, bs_index: int, bs_type: str, bs_loc, bs_radius: float):
        self.sce = sce
        self.id = bs_index
        self.bs_type = bs_type
        self._loc = bs_loc
        self.radius = bs_radius
        self._ch_state = np.zeros(self.sce.nChannel)

    # --- state / geometry ---
    def reset(self):
        self._ch_state = np.zeros(self.sce.nChannel)

    def get_location(self):
        return self._loc

    # --- RF characteristics ---
    def transmit_power_dbm(self) -> float:
        return TX_POWER_DBM[self.bs_type]

    def receive_power(self, d: float) -> float:
        """Calculate received power in mW using simplified path loss.

        Args:
            d: Distance in meters
        Returns:
            Received power in milliwatts (linear scale). 0.0 if outside
            coverage radius.
        """
        d = max(d, 1.0)  # Avoid log10(0)
        tx_power_dbm = self.transmit_power_dbm()
        if self.bs_type in ("MBS", "PBS"):
            loss = 34 + 40 * np.log10(d)
        else:  # FBS
            loss = 37 + 30 * np.log10(d)
        if d <= self.radius:
            rx_power_dbm = tx_power_dbm - loss
            return 10 ** (rx_power_dbm / 10)
        return 0.0

    # ---- Legacy API compatibility (old camelCase method names) ----
    def Get_Location(self):  # noqa: N802
        return self.get_location()

    def Transmit_Power_dBm(self):  # noqa: N802
        return self.transmit_power_dbm()

    def Receive_Power(self, d):  # noqa: N802
        return self.receive_power(d)


__all__ = ["BS", "TX_POWER_DBM"]
