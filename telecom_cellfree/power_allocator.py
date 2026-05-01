"""Power allocation utilities for Cell-Free serving clusters."""

from __future__ import annotations

from typing import Dict, Iterable


class ClusterPowerAllocator:
    """Simple AP-cluster power allocation with profile scaling."""

    def __init__(self, max_ap_power_dbm: float):
        self.max_ap_power_dbm = float(max_ap_power_dbm)

    @staticmethod
    def dbm_to_mw(power_dbm: float) -> float:
        return 10 ** (float(power_dbm) / 10.0)

    def allocate(
        self,
        cluster_ap_ids: Iterable[int],
        profile_scale: float,
        ap_user_count: Dict[int, int],
    ) -> Dict[int, float]:
        """Allocate mW per AP for one UE.

        Each AP power is scaled by profile_scale and shared across associated UEs.
        """
        allocation = {}
        max_power_mw = self.dbm_to_mw(self.max_ap_power_dbm)
        scale = max(0.05, min(1.0, float(profile_scale)))

        for ap_id in cluster_ap_ids:
            users = max(1, int(ap_user_count.get(int(ap_id), 1)))
            allocation[int(ap_id)] = (max_power_mw * scale) / users

        return allocation
