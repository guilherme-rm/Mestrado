"""AP-UE association helpers for user-centric Cell-Free clustering."""

from __future__ import annotations

from typing import List
import numpy as np


class UserCentricAssociationManager:
    """Builds serving AP clusters using strongest large-scale fading links."""

    def __init__(self, serving_aps_per_user: int):
        self.serving_aps_per_user = max(1, int(serving_aps_per_user))

    def strongest_cluster(self, large_scale_fading: np.ndarray) -> List[int]:
        """Return top-N AP indices by large-scale fading strength."""
        if large_scale_fading.size == 0:
            return []
        k = min(self.serving_aps_per_user, large_scale_fading.size)
        order = np.argsort(-large_scale_fading)
        return [int(i) for i in order[:k]]

    def cluster_from_primary(self, primary_ap: int, large_scale_fading: np.ndarray) -> List[int]:
        """Return cluster containing primary AP plus strongest neighboring APs."""
        if large_scale_fading.size == 0:
            return []

        primary_ap = int(np.clip(primary_ap, 0, large_scale_fading.size - 1))
        order = np.argsort(-large_scale_fading)

        cluster = [primary_ap]
        for ap_idx in order:
            ap_idx = int(ap_idx)
            if ap_idx == primary_ap:
                continue
            cluster.append(ap_idx)
            if len(cluster) >= self.serving_aps_per_user:
                break

        return cluster
