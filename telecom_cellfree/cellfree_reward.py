"""Reward model for user-centric Cell-Free Massive MIMO."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import torch

from .association_manager import UserCentricAssociationManager
from .power_allocator import ClusterPowerAllocator


class CellFreeRewardCalculator:
    """Cell-Free reward and QoS evaluator with cluster-based serving."""

    def __init__(self, sce, opt, device: torch.device):
        self.sce = sce
        self.opt = opt
        self.device = device
        self._last_clusters: Dict[int, List[int]] = {}

    def compute(
        self,
        action_all: torch.Tensor,
        action_i: torch.Tensor,
        agent_location: torch.Tensor,
        scenario,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        aps = scenario.Get_BaseStations()
        n_ap = len(aps)
        if n_ap == 0:
            return self._invalid(scenario)

        n_channel = int(getattr(self.sce, "nChannel", 1) or 1)
        action_idx = self._action_to_int(action_i)

        primary_ap = int(action_idx // n_channel)
        power_profile = int(action_idx % n_channel)
        primary_ap = int(np.clip(primary_ap, 0, n_ap - 1))

        ue_pos_np = self._to_numpy(agent_location)
        beta_self = scenario.large_scale_fading(ue_pos_np)

        assoc_manager = UserCentricAssociationManager(
            getattr(scenario.cfg, "serving_aps_per_user", getattr(self.sce, "serving_aps_per_user", 4))
        )
        serving_cluster = assoc_manager.cluster_from_primary(primary_ap, beta_self)

        # Track AP associations/load for diagnostics and future extensions.
        ap_user_count = self._estimate_ap_user_count(action_all, scenario, assoc_manager)
        for ap_idx in serving_cluster:
            aps[ap_idx].current_load = float(ap_user_count.get(ap_idx, 1))

        cpu_loads = self._estimate_cpu_loads(scenario, ap_user_count)
        if hasattr(scenario, "set_visual_diagnostics"):
            scenario.set_visual_diagnostics(
                serving_clusters=self._last_clusters,
                ap_loads=ap_user_count,
                cpu_loads=cpu_loads,
            )

        profile_scale = scenario.get_power_profile_scale(power_profile)
        allocator = ClusterPowerAllocator(getattr(scenario.cfg, "ap_max_tx_power_dbm", 23.0))
        per_ap_power = allocator.allocate(serving_cluster, profile_scale, ap_user_count)

        useful_signal = self._compute_useful_signal(serving_cluster, per_ap_power, beta_self, scenario)
        interference = self._compute_interference(
            action_all=action_all,
            target_ue_pos=ue_pos_np,
            scenario=scenario,
            assoc_manager=assoc_manager,
            exclude_cluster=set(serving_cluster),
            ap_user_count=ap_user_count,
        )

        noise_mw = (10 ** (float(self.sce.N0) / 10.0)) * float(self.sce.BW)
        sinr = useful_signal / max(interference + noise_mw, 1e-20)
        rate_mbps = (float(self.sce.BW) * np.log2(1.0 + sinr)) / 1e6

        min_required_rate = float(getattr(self.sce, "min_required_rate_mbps", 1.0) or 1.0)
        qos = 1 if rate_mbps >= min_required_rate else 0

        total_power_mw = float(sum(per_ap_power.values()))
        profit = float(getattr(self.sce, "profit", 1.0) or 1.0)
        power_cost = float(getattr(self.sce, "power_cost", 0.0005) or 0.0005)
        action_cost = float(getattr(self.sce, "action_cost", 0.001) or 0.001)
        qos_penalty = float(getattr(self.sce, "qos_penalty", 0.5) or 0.5)

        reward = profit * rate_mbps - power_cost * total_power_mw - action_cost
        if qos == 0:
            reward -= qos_penalty

        return (
            qos,
            torch.tensor(reward, device=self.device, dtype=torch.float32),
            torch.tensor(rate_mbps, device=self.device, dtype=torch.float32),
        )

    def _invalid(self, scenario):
        negative = float(getattr(self.sce, "negative_cost", -0.5) or -0.5)
        return (
            0,
            torch.tensor(negative, device=self.device, dtype=torch.float32),
            torch.tensor(0.0, device=self.device, dtype=torch.float32),
        )

    def _estimate_ap_user_count(self, action_all, scenario, assoc_manager) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        n_channel = int(getattr(self.sce, "nChannel", 1) or 1)
        ue_positions = scenario.get_current_ue_positions() or []

        for ue_idx, action in enumerate(action_all):
            action_idx = self._action_to_int(action)
            primary_ap = int(np.clip(action_idx // n_channel, 0, scenario.BS_Number() - 1))

            if ue_idx < len(ue_positions):
                beta = scenario.large_scale_fading(ue_positions[ue_idx])
            else:
                beta = np.ones(scenario.BS_Number(), dtype=np.float64)

            cluster = assoc_manager.cluster_from_primary(primary_ap, beta)
            self._last_clusters[ue_idx] = cluster
            for ap_idx in cluster:
                counts[ap_idx] = counts.get(ap_idx, 0) + 1

        return counts

    def _compute_useful_signal(
        self,
        serving_cluster: List[int],
        per_ap_power: Dict[int, float],
        beta: np.ndarray,
        scenario,
    ) -> float:
        if not serving_cluster:
            return 0.0

        h2 = scenario.sample_small_scale_power(len(serving_cluster))
        total = 0.0
        for i, ap_idx in enumerate(serving_cluster):
            total += float(per_ap_power[ap_idx]) * float(beta[ap_idx]) * float(h2[i])
        return max(total, 0.0)

    def _compute_interference(
        self,
        action_all,
        target_ue_pos: np.ndarray,
        scenario,
        assoc_manager,
        exclude_cluster,
        ap_user_count: Dict[int, int],
    ) -> float:
        n_channel = int(getattr(self.sce, "nChannel", 1) or 1)
        interference = 0.0

        for ue_idx, action in enumerate(action_all):
            action_idx = self._action_to_int(action)
            profile_idx = int(action_idx % n_channel)
            profile_scale = scenario.get_power_profile_scale(profile_idx)

            cluster = self._last_clusters.get(ue_idx)
            if cluster is None:
                primary_ap = int(np.clip(action_idx // n_channel, 0, scenario.BS_Number() - 1))
                beta_ue = scenario.large_scale_fading(target_ue_pos)
                cluster = assoc_manager.cluster_from_primary(primary_ap, beta_ue)

            interfering_aps = [ap for ap in cluster if ap not in exclude_cluster]
            if not interfering_aps:
                continue

            allocator = ClusterPowerAllocator(getattr(scenario.cfg, "ap_max_tx_power_dbm", 23.0))
            power_map = allocator.allocate(interfering_aps, profile_scale, ap_user_count)

            beta_target = scenario.large_scale_fading(target_ue_pos)
            h2 = scenario.sample_small_scale_power(len(interfering_aps))
            for i, ap_idx in enumerate(interfering_aps):
                interference += float(power_map[ap_idx]) * float(beta_target[ap_idx]) * float(h2[i])

        return max(interference, 0.0)

    def _estimate_cpu_loads(self, scenario, ap_user_count: Dict[int, int]) -> Dict[int, float]:
        """Compute normalized CPU load proxy from AP association counts."""
        cpu_loads: Dict[int, float] = {}
        cpus = scenario.get_cpus() if hasattr(scenario, "get_cpus") else []
        for cpu in cpus:
            total_users = sum(ap_user_count.get(int(ap_id), 0) for ap_id in cpu.connected_ap_ids)
            capacity = max(1.0, float(getattr(cpu, "fronthaul_capacity_mbps", 10000.0)))
            # Proxy throughput demand in Mbps (rough visualization metric)
            demand_mbps = total_users * float(getattr(self.sce, "min_required_rate_mbps", 1.0) or 1.0)
            cpu_loads[int(cpu.id)] = min(1.0, demand_mbps / capacity)
        return cpu_loads

    @staticmethod
    def _to_numpy(x: torch.Tensor) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x, dtype=np.float32)

    @staticmethod
    def _action_to_int(action) -> int:
        if isinstance(action, torch.Tensor):
            return int(action.detach().item())
        return int(action)
