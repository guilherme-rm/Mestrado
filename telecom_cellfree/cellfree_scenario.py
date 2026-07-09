"""Cell-Free Massive MIMO scenario and channel modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence, Tuple
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
    ap_layout: str
    ap_grid_jitter_ratio: float
    enforce_user_count: bool
    target_nagents: int
    min_nagents: int
    max_nagents: int
    interference_scale: float
    curriculum_enabled: bool
    curriculum_min_rate_factor_start: float
    curriculum_interference_scale_start: float

    @classmethod
    def from_sce(cls, sce):
        legacy_ap_count = (
            int(getattr(sce, "nMBS", 0) or 0)
            + int(getattr(sce, "nPBS", 0) or 0)
            + int(getattr(sce, "nFBS", 0) or 0)
        )

        n_ap = int(getattr(sce, "nAP", 0) or legacy_ap_count or 32)
        target_nagents = int(getattr(sce, "target_nagents", 0) or 0)
        min_nagents = int(getattr(sce, "min_nagents", 0) or 0)
        max_nagents = int(getattr(sce, "max_nagents", 0) or 0)
        if target_nagents <= 0:
            target_nagents = max(1, int(round(n_ap / 4)))
        if min_nagents <= 0:
            min_nagents = max(1, int(round(target_nagents * 0.7)))
        if max_nagents <= 0:
            max_nagents = max(min_nagents, int(round(target_nagents * 1.3)))
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
            ap_layout=str(getattr(sce, "ap_layout", "grid_jitter") or "grid_jitter").strip().lower(),
            ap_grid_jitter_ratio=float(getattr(sce, "ap_grid_jitter_ratio", 0.25) or 0.25),
            enforce_user_count=bool(getattr(sce, "enforce_user_count", True)),
            target_nagents=target_nagents,
            min_nagents=min_nagents,
            max_nagents=max_nagents,
            interference_scale=float(getattr(sce, "interference_scale", 0.25) or 0.25),
            curriculum_enabled=bool(getattr(sce, "curriculum_enabled", True)),
            curriculum_min_rate_factor_start=float(getattr(sce, "curriculum_min_rate_factor_start", 0.6) or 0.6),
            curriculum_interference_scale_start=float(getattr(sce, "curriculum_interference_scale_start", 0.5) or 0.5),
        )


class CellFreeScenario:
    """Cell-Free topology with distributed APs and optional CPU grouping."""

    def __init__(self, sce):
        self.sce = sce
        self.cfg = CellFreeScenarioConfig.from_sce(sce)
        self._step_id = 0
        self._curriculum_progress = 1.0
        self._aps = self._init_access_points()
        self._cpus = self._init_cpus()
        self._current_ue_positions: Optional[List[np.ndarray]] = None
        self._step_large_scale_cache: Dict[Tuple[float, float], np.ndarray] = {}
        self._step_small_scale_cache: Dict[Hashable, np.ndarray] = {}
        self._visual_diagnostics = {
            "serving_clusters": {},
            "ap_loads": {},
            "cpu_loads": {},
            "cpu_positions": self._compute_cpu_positions(),
            "cpu_ap_links": {
                int(cpu.id): [int(ap) for ap in cpu.connected_ap_ids]
                for cpu in self._cpus
            },
            "step_metrics": {},
        }

    def reset(self):
        for ap in self._aps:
            ap.reset()
        for cpu in self._cpus:
            cpu.reset()
        self.begin_step()

    def begin_step(self):
        """Reset per-step channel caches to keep one realization per transition."""
        self._step_id += 1
        self._step_large_scale_cache.clear()
        self._step_small_scale_cache.clear()

    def step_id(self) -> int:
        return self._step_id

    def set_curriculum_progress(self, episode_idx: int, total_episodes: int):
        if not self.cfg.curriculum_enabled:
            self._curriculum_progress = 1.0
            return
        denom = max(1, int(total_episodes) - 1)
        self._curriculum_progress = float(np.clip(float(episode_idx) / float(denom), 0.0, 1.0))

    def get_effective_interference_scale(self) -> float:
        start = float(self.cfg.curriculum_interference_scale_start)
        end = float(self.cfg.interference_scale)
        p = self._curriculum_progress
        return float(start + (end - start) * p)

    def get_effective_min_rate_factor(self) -> float:
        start = float(self.cfg.curriculum_min_rate_factor_start)
        end = 1.0
        p = self._curriculum_progress
        return float(start + (end - start) * p)

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
        step_metrics: Optional[Dict[str, object]] = None,
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
        self._visual_diagnostics["step_metrics"] = dict(step_metrics or {})

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
        cache_key = tuple(np.round(ue_pos.astype(np.float64), 4))
        cached = self._step_large_scale_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

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
        beta = np.maximum(beta, 1e-16)
        self._step_large_scale_cache[cache_key] = beta.copy()
        return beta

    def sample_small_scale_power(self, n_links: int, cache_key: Hashable | None = None) -> np.ndarray:
        """Sample |h_lk|^2 with Rayleigh fading (exponential unit mean)."""
        if cache_key is not None:
            cached = self._step_small_scale_cache.get(cache_key)
            if cached is not None:
                return cached.copy()

        samples = np.random.exponential(scale=1.0, size=int(n_links))
        if cache_key is not None:
            self._step_small_scale_cache[cache_key] = samples.copy()
        return samples

    def _init_access_points(self) -> List[AccessPoint]:
        aps = []
        if self.cfg.ap_layout in {"grid", "grid_jitter"}:
            positions = self._generate_grid_positions(
                self.cfg.n_ap,
                self.cfg.area_width,
                self.cfg.area_height,
                jitter_ratio=self.cfg.ap_grid_jitter_ratio if self.cfg.ap_layout == "grid_jitter" else 0.0,
            )
        else:
            positions = [
                np.array([
                    np.random.uniform(0.0, self.cfg.area_width),
                    np.random.uniform(0.0, self.cfg.area_height),
                ], dtype=np.float32)
                for _ in range(self.cfg.n_ap)
            ]

        for ap_id, pos in enumerate(positions):
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

    def _generate_grid_positions(
        self,
        n_ap: int,
        area_width: float,
        area_height: float,
        jitter_ratio: float,
    ) -> List[np.ndarray]:
        n_ap = max(1, int(n_ap))
        n_cols = int(np.ceil(np.sqrt(n_ap * area_width / max(area_height, 1e-6))))
        n_rows = int(np.ceil(n_ap / max(1, n_cols)))
        xs = np.linspace(0.0, area_width, n_cols + 2)[1:-1]
        ys = np.linspace(0.0, area_height, n_rows + 2)[1:-1]
        dx = float(area_width) / float(max(1, n_cols + 1))
        dy = float(area_height) / float(max(1, n_rows + 1))
        jx = float(jitter_ratio) * dx
        jy = float(jitter_ratio) * dy

        out: List[np.ndarray] = []
        for y in ys:
            for x in xs:
                if len(out) >= n_ap:
                    break
                px = float(np.clip(x + np.random.uniform(-jx, jx), 0.0, area_width))
                py = float(np.clip(y + np.random.uniform(-jy, jy), 0.0, area_height))
                out.append(np.array([px, py], dtype=np.float32))
            if len(out) >= n_ap:
                break
        return out

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
