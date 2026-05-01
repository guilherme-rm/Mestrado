"""Cell-Free Massive MIMO scenario and channel modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np

from .access_point import AccessPoint
from .cpu import CPU


@dataclass
class CellFreeScenarioConfig:
    """Typed config helper for Cell-Free parameters."""

    n_ap: int
    n_cpu: int
    area_width: float
    area_height: float
    ap_max_tx_power_dbm: float
    ap_coverage_radius: float
    serving_aps_per_user: int
    shadowing_sigma_db: float
    min_distance: float

    @classmethod
    def from_sce(cls, sce):
        legacy_ap_count = (
            int(getattr(sce, "nMBS", 0) or 0)
            + int(getattr(sce, "nPBS", 0) or 0)
            + int(getattr(sce, "nFBS", 0) or 0)
        )

        n_ap = int(getattr(sce, "nAP", 0) or legacy_ap_count or 32)
        return cls(
            n_ap=n_ap,
            n_cpu=int(getattr(sce, "nCPU", 0) or 1),
            area_width=float(getattr(sce, "area_width", 2000.0) or 2000.0),
            area_height=float(getattr(sce, "area_height", 1000.0) or 1000.0),
            ap_max_tx_power_dbm=float(getattr(sce, "ap_max_tx_power_dbm", 23.0) or 23.0),
            ap_coverage_radius=float(getattr(sce, "ap_coverage_radius", 500.0) or 500.0),
            serving_aps_per_user=int(getattr(sce, "serving_aps_per_user", 4) or 4),
            shadowing_sigma_db=float(getattr(sce, "shadowing_sigma_db", 0.0) or 0.0),
            min_distance=float(getattr(sce, "min_distance", 1.0) or 1.0),
        )


class CellFreeScenario:
    """Cell-Free topology with distributed APs and optional CPU grouping."""

    def __init__(self, sce):
        self.sce = sce
        self.cfg = CellFreeScenarioConfig.from_sce(sce)
        self._aps = self._init_access_points()
        self._cpus = self._init_cpus()
        self._current_ue_positions: Optional[List[np.ndarray]] = None
        self._visual_diagnostics = {
            "serving_clusters": {},
            "ap_loads": {},
            "cpu_loads": {},
            "cpu_positions": self._compute_cpu_positions(),
            "cpu_ap_links": {
                int(cpu.id): [int(ap) for ap in cpu.connected_ap_ids]
                for cpu in self._cpus
            },
        }

    def reset(self):
        for ap in self._aps:
            ap.reset()
        for cpu in self._cpus:
            cpu.reset()

    def Get_BaseStations(self):  # noqa: N802
        return self._aps

    def BS_Number(self):  # noqa: N802
        return len(self._aps)

    def BS_Location(self):  # noqa: N802
        loc_ap = np.array([ap.get_location() for ap in self._aps], dtype=np.float32)
        return loc_ap, np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    def get_cpus(self) -> List[CPU]:
        return self._cpus

    def set_current_ue_positions(self, positions: Sequence[np.ndarray]):
        self._current_ue_positions = [np.array(p, dtype=np.float32) for p in positions]

    def get_current_ue_positions(self) -> Optional[List[np.ndarray]]:
        return self._current_ue_positions

    def set_visual_diagnostics(
        self,
        serving_clusters,
        ap_loads,
        cpu_loads,
    ):
        self._visual_diagnostics["serving_clusters"] = {
            int(k): [int(ap) for ap in v]
            for k, v in serving_clusters.items()
        }
        self._visual_diagnostics["ap_loads"] = {
            int(k): float(v)
            for k, v in ap_loads.items()
        }
        self._visual_diagnostics["cpu_loads"] = {
            int(k): float(v)
            for k, v in cpu_loads.items()
        }

    def get_visual_diagnostics(self):
        return self._visual_diagnostics

    def get_power_profile_scale(self, profile_idx: int) -> float:
        profile_values = getattr(self.sce, "power_profile_scales", None)
        if profile_values is not None and len(profile_values) > 0:
            idx = int(np.clip(profile_idx, 0, len(profile_values) - 1))
            return float(profile_values[idx])

        n_profiles = int(getattr(self.sce, "nChannel", 4) or 4)
        n_profiles = max(1, n_profiles)
        values = np.linspace(0.4, 1.0, n_profiles)
        idx = int(np.clip(profile_idx, 0, n_profiles - 1))
        return float(values[idx])

    def large_scale_fading(self, ue_position: np.ndarray) -> np.ndarray:
        """Compute beta_lk from distance-based path loss and optional shadowing."""
        ue_pos = np.array(ue_position, dtype=np.float32)
        ap_pos = np.array([ap.get_location() for ap in self._aps], dtype=np.float32)

        d = np.linalg.norm(ap_pos - ue_pos[None, :], axis=1)
        d = np.maximum(d, self.cfg.min_distance)

        # Typical urban micro path-loss expression (d in meters).
        path_loss_db = 30.5 + 36.7 * np.log10(d)

        sigma = self.cfg.shadowing_sigma_db
        if sigma > 0:
            shadow_db = np.random.normal(0.0, sigma, size=path_loss_db.shape)
            path_loss_db = path_loss_db - shadow_db

        beta = 10 ** (-path_loss_db / 10.0)
        return np.maximum(beta, 1e-16)

    def sample_small_scale_power(self, n_links: int) -> np.ndarray:
        """Sample |h_lk|^2 with Rayleigh fading (exponential unit mean)."""
        return np.random.exponential(scale=1.0, size=int(n_links))

    def _init_access_points(self) -> List[AccessPoint]:
        aps = []
        for ap_id in range(self.cfg.n_ap):
            pos = np.array([
                np.random.uniform(0.0, self.cfg.area_width),
                np.random.uniform(0.0, self.cfg.area_height),
            ], dtype=np.float32)
            aps.append(
                AccessPoint(
                    id=ap_id,
                    position=pos,
                    max_tx_power_dbm=self.cfg.ap_max_tx_power_dbm,
                    radius=self.cfg.ap_coverage_radius,
                    antenna_count=int(getattr(self.sce, "ap_antenna_count", 1) or 1),
                    cpu_id=(ap_id % max(1, self.cfg.n_cpu)),
                )
            )
        return aps

    def _init_cpus(self) -> List[CPU]:
        n_cpu = max(1, self.cfg.n_cpu)
        cpus = [
            CPU(
                id=i,
                connected_ap_ids=[],
                fronthaul_capacity_mbps=float(
                    getattr(self.sce, "fronthaul_capacity_mbps", 10000.0) or 10000.0
                ),
            )
            for i in range(n_cpu)
        ]

        for ap in self._aps:
            cpus[ap.cpu_id].connected_ap_ids.append(ap.id)

        return cpus

    def _compute_cpu_positions(self):
        """Place CPU markers near AP centroids for topology visualization."""
        positions = {}
        for cpu in self._cpus:
            ap_ids = cpu.connected_ap_ids
            if not ap_ids:
                positions[int(cpu.id)] = np.array([0.0, 0.0], dtype=np.float32)
                continue
            ap_pos = np.array([self._aps[i].get_location() for i in ap_ids], dtype=np.float32)
            centroid = ap_pos.mean(axis=0)
            positions[int(cpu.id)] = centroid
        return positions
