"""Scenario definition containing base station placement and geometry."""

from __future__ import annotations

import numpy as np
from numpy import pi
from random import random, uniform, choice
from typing import List, Tuple

from dataclasses import dataclass

from .base_station import BS

@dataclass
class ScenarioConfig:
    """Typed configuration for scenario parameters."""
    nMBS: int
    nPBS: int
    nFBS: int
    rMBS: float
    rPBS: float
    rFBS: float
    BW: float
    nChannel: int
    N0: float
    QoS_thr: float
    profit: float = 1.0
    power_cost: float = 0.0005
    action_cost: float = 0.001
    negative_cost: float = -0.5
    
    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioConfig":
        """Create config from dictionary (e.g., loaded JSON)."""
        return cls(
            nMBS=d["nMBS"],
            nPBS=d["nPBS"],
            nFBS=d["nFBS"],
            rMBS=d["rMBS"],
            rPBS=d["rPBS"],
            rFBS=d["rFBS"],
            BW=d["BW"],
            nChannel=d["nChannel"],
            N0=d["N0"],
            QoS_thr=d["QoS_thr"],
            profit=d.get("profit", 1.0),
            power_cost=d.get("power_cost", 0.0005),
            action_cost=d.get("action_cost", 0.001),
            negative_cost=d.get("negative_cost", -0.5),
        )
    
    @property
    def total_bs(self) -> int:
        """Total number of base stations."""
        return self.nMBS + self.nPBS + self.nFBS

class Scenario:
    """Defines the network scenario (layout + base stations)."""

    def __init__(self, sce):
        self.sce = sce
        self._base_stations = self._init_base_stations()

    # Compatibility alias
    def Get_BaseStations(self):  # noqa: N802 (legacy name)
        return self._base_stations

    def BS_Number(self) -> int:  # noqa: N802 (legacy name)
        return self.sce.nMBS + self.sce.nPBS + self.sce.nFBS

    def BS_Location(self):  # noqa: N802 (legacy name)
        return self._bs_location()

    def reset(self):
        for bs in self._base_stations:
            bs.reset()

    # --- internal helpers ---
    def _bs_location(self):
        loc_mbs = np.zeros((self.sce.nMBS, 2))
        loc_pbs = np.zeros((self.sce.nPBS, 2))
        loc_fbs = np.zeros((self.sce.nFBS, 2))

        for i in range(self.sce.nMBS):
            loc_mbs[i, 0] = 500 + 900 * i
            loc_mbs[i, 1] = 500

        for i in range(self.sce.nPBS):
            loc_pbs[i, 0] = loc_mbs[int(i / 4), 0] + 250 * np.cos(pi / 2 * (i % 4))
            loc_pbs[i, 1] = loc_mbs[int(i / 4), 1] + 250 * np.sin(pi / 2 * (i % 4))

        for i in range(self.sce.nFBS):
            locm = choice(loc_mbs)
            r = self.sce.rMBS * random()
            theta = uniform(-pi, pi)
            loc_fbs[i, 0] = locm[0] + r * np.cos(theta)
            loc_fbs[i, 1] = locm[1] + r * np.sin(theta)

        return loc_mbs, loc_pbs, loc_fbs

    def _init_base_stations(self) -> List[BS]:
        stations: List[BS] = []
        loc_mbs, loc_pbs, loc_fbs = self._bs_location()

        for i in range(self.sce.nMBS):
            stations.append(BS(self.sce, i, "MBS", loc_mbs[i], self.sce.rMBS))

        for i in range(self.sce.nPBS):
            idx = self.sce.nMBS + i
            stations.append(BS(self.sce, idx, "PBS", loc_pbs[i], self.sce.rPBS))

        for i in range(self.sce.nFBS):
            idx = self.sce.nMBS + self.sce.nPBS + i
            stations.append(BS(self.sce, idx, "FBS", loc_fbs[i], self.sce.rFBS))

        return stations


__all__ = ["Scenario"]
