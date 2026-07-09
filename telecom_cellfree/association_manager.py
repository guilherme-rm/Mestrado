"""AP-UE association helpers for user-centric Cell-Free clustering."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class UserCentricAssociationManager:
    """Build serving clusters with configurable user-centric policies."""

    def __init__(
        self,
        serving_aps_per_user: int,
        strategy: str = "topn",
        distance_weight: float = 0.6,
        gain_weight: float = 0.35,
        load_weight: float = 0.05,
        dynamic_size: bool = False,
        min_cluster_size: Optional[int] = None,
        max_cluster_size: Optional[int] = None,
    ):
        self.serving_aps_per_user = max(1, int(serving_aps_per_user))
        self.strategy = str(strategy or "topn").strip().lower()
        self.distance_weight = float(max(0.0, distance_weight))
        self.gain_weight = float(max(0.0, gain_weight))
        self.load_weight = float(max(0.0, load_weight))
        self.dynamic_size = bool(dynamic_size)
        self.min_cluster_size = max(1, int(min_cluster_size or self.serving_aps_per_user))
        self.max_cluster_size = max(self.min_cluster_size, int(max_cluster_size or self.serving_aps_per_user))

    def strongest_cluster(self, large_scale_fading: np.ndarray) -> List[int]:
        """Return top-N AP indices by large-scale fading strength."""
        if large_scale_fading.size == 0:
            return []
        k = min(self.serving_aps_per_user, large_scale_fading.size)
        order = np.argsort(-large_scale_fading)
        return [int(i) for i in order[:k]]

    def cluster_from_primary(
        self,
        primary_ap: int,
        large_scale_fading: np.ndarray,
        ap_distances: Optional[np.ndarray] = None,
        ap_user_count: Optional[Dict[int, int]] = None,
    ) -> List[int]:
        """Return cluster containing primary AP and selected cooperating APs."""
        if large_scale_fading.size == 0:
            return []

        primary_ap = int(np.clip(primary_ap, 0, large_scale_fading.size - 1))

        cluster_size = self._resolve_cluster_size(large_scale_fading, primary_ap)
        if self.strategy in {"topn", "strongest"}:
            order = np.argsort(-large_scale_fading)
        elif self.strategy in {"distance", "distance_aware", "hybrid", "load_aware"}:
            order = self._rank_with_composite_score(
                large_scale_fading,
                ap_distances=ap_distances,
                ap_user_count=ap_user_count,
            )
        else:
            order = np.argsort(-large_scale_fading)

        cluster = [primary_ap]
        for ap_idx in order:
            ap_idx = int(ap_idx)
            if ap_idx == primary_ap:
                continue
            cluster.append(ap_idx)
            if len(cluster) >= cluster_size:
                break

        return cluster

    def _resolve_cluster_size(self, large_scale_fading: np.ndarray, primary_ap: int) -> int:
        if not self.dynamic_size:
            return min(self.serving_aps_per_user, int(large_scale_fading.size))

        strongest = float(np.max(large_scale_fading))
        primary_gain = float(large_scale_fading[primary_ap])
        quality_ratio = primary_gain / max(strongest, 1e-12)
        # Weaker primary links trigger larger clusters for diversity.
        expand = int(round((1.0 - quality_ratio) * (self.max_cluster_size - self.min_cluster_size)))
        target = self.min_cluster_size + max(0, expand)
        return max(1, min(int(large_scale_fading.size), target))

    def _rank_with_composite_score(
        self,
        large_scale_fading: np.ndarray,
        ap_distances: Optional[np.ndarray],
        ap_user_count: Optional[Dict[int, int]],
    ) -> np.ndarray:
        n_ap = int(large_scale_fading.size)
        gain = np.array(large_scale_fading, dtype=np.float64)
        gain_norm = gain / max(float(np.max(gain)), 1e-12)

        if ap_distances is None or len(ap_distances) != n_ap:
            dist_norm = np.ones(n_ap, dtype=np.float64)
        else:
            dist = np.array(ap_distances, dtype=np.float64)
            dist = np.maximum(dist, 1e-6)
            inv_dist = 1.0 / dist
            dist_norm = inv_dist / max(float(np.max(inv_dist)), 1e-12)

        if ap_user_count is None:
            load_norm = np.zeros(n_ap, dtype=np.float64)
        else:
            load = np.array([float(ap_user_count.get(i, 0)) for i in range(n_ap)], dtype=np.float64)
            load_norm = load / max(float(np.max(load)), 1.0)

        denom = max(self.distance_weight + self.gain_weight + self.load_weight, 1e-12)
        score = (
            self.distance_weight * dist_norm
            + self.gain_weight * gain_norm
            - self.load_weight * load_norm
        ) / denom
        return np.argsort(-score)
