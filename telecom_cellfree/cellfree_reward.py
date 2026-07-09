"""Reward model for user-centric Cell-Free Massive MIMO."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
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
        self._step_cache: Dict[str, object] = {}
        self._last_step_id: int | None = None

    def compute(
        self,
        action_all: torch.Tensor,
        action_i: torch.Tensor,
        agent_location: torch.Tensor,
        scenario,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        step_id = int(scenario.step_id()) if hasattr(scenario, "step_id") else -1
        if self._last_step_id != step_id:
            self._step_cache = self._compute_step_snapshot(action_all, scenario)
            self._last_step_id = step_id

        ue_idx = self._infer_ue_index(agent_location, scenario)
        if ue_idx < 0:
            return self._invalid(scenario)

        rates = self._step_cache["rates_mbps"]
        rewards = self._step_cache["rewards"]
        binary_qos = self._step_cache["binary_qos"]
        if ue_idx >= len(rates):
            return self._invalid(scenario)

        return (
            int(binary_qos[ue_idx]),
            torch.tensor(float(rewards[ue_idx]), device=self.device, dtype=torch.float32),
            torch.tensor(float(rates[ue_idx]), device=self.device, dtype=torch.float32),
        )

    def _invalid(self, scenario):
        negative = float(getattr(self.sce, "negative_cost", -0.5) or -0.5)
        return (
            0,
            torch.tensor(negative, device=self.device, dtype=torch.float32),
            torch.tensor(0.0, device=self.device, dtype=torch.float32),
        )

    def _compute_step_snapshot(self, action_all, scenario) -> Dict[str, object]:
        aps = scenario.Get_BaseStations()
        n_ap = len(aps)
        if n_ap <= 0:
            return {
                "rates_mbps": np.zeros(0, dtype=np.float64),
                "rewards": np.zeros(0, dtype=np.float64),
                "binary_qos": np.zeros(0, dtype=np.int64),
            }

        n_users = int(len(action_all))
        n_channel = int(getattr(self.sce, "nChannel", 1) or 1)
        ue_positions = scenario.get_current_ue_positions() or []
        if len(ue_positions) < n_users:
            ue_positions = list(ue_positions) + [np.zeros(2, dtype=np.float32) for _ in range(n_users - len(ue_positions))]

        min_required_rate = self._required_rate_mbps(scenario)
        assoc_manager = UserCentricAssociationManager(
            getattr(scenario.cfg, "serving_aps_per_user", getattr(self.sce, "serving_aps_per_user", 4)),
            strategy=str(getattr(self.sce, "cluster_assignment_mode", "hybrid") or "hybrid"),
            distance_weight=float(getattr(self.sce, "cluster_distance_weight", 0.6) or 0.6),
            gain_weight=float(getattr(self.sce, "cluster_gain_weight", 0.35) or 0.35),
            load_weight=float(getattr(self.sce, "cluster_load_weight", 0.05) or 0.05),
            dynamic_size=bool(getattr(self.sce, "dynamic_cluster_sizing", False)),
            min_cluster_size=int(getattr(self.sce, "min_serving_aps_per_user", getattr(self.sce, "serving_aps_per_user", 4)) or 1),
            max_cluster_size=int(getattr(self.sce, "max_serving_aps_per_user", getattr(self.sce, "serving_aps_per_user", 4)) or 1),
        )

        action_idx = np.array([self._action_to_int(a) for a in action_all], dtype=np.int64)
        primary_ap = np.clip(action_idx // max(1, n_channel), 0, max(0, n_ap - 1)).astype(np.int64)
        profile_idx = (action_idx % max(1, n_channel)).astype(np.int64)

        beta_all = [scenario.large_scale_fading(np.array(ue_positions[i], dtype=np.float32)) for i in range(n_users)]
        distances_all = [self._compute_ap_distances(aps, np.array(ue_positions[i], dtype=np.float32)) for i in range(n_users)]

        # First pass cluster selection using baseline load estimate.
        ap_user_count: Dict[int, int] = {}
        serving_clusters: Dict[int, List[int]] = {}
        for ue_idx in range(n_users):
            cluster = assoc_manager.cluster_from_primary(
                int(primary_ap[ue_idx]),
                beta_all[ue_idx],
                ap_distances=distances_all[ue_idx],
                ap_user_count=ap_user_count,
            )
            serving_clusters[ue_idx] = cluster
            for ap_idx in cluster:
                ap_user_count[ap_idx] = ap_user_count.get(ap_idx, 0) + 1

        active_users = self._sample_active_users(n_users)
        active_mask = np.array([i in active_users for i in range(n_users)], dtype=bool)
        per_ue_power_map: Dict[int, Dict[int, float]] = {}
        allocator = ClusterPowerAllocator(getattr(scenario.cfg, "ap_max_tx_power_dbm", 23.0))
        for ue_idx in range(n_users):
            if ue_idx not in active_users:
                per_ue_power_map[ue_idx] = {}
                continue
            scale = scenario.get_power_profile_scale(int(profile_idx[ue_idx]))
            per_ue_power_map[ue_idx] = allocator.allocate(serving_clusters[ue_idx], scale, ap_user_count)

        noise_mw = (10 ** (float(self.sce.N0) / 10.0)) * float(self.sce.BW)
        effective_interference_scale = (
            float(scenario.get_effective_interference_scale())
            if hasattr(scenario, "get_effective_interference_scale")
            else float(getattr(self.sce, "interference_scale", 0.25) or 0.25)
        )

        signal = np.zeros(n_users, dtype=np.float64)
        interference = np.zeros(n_users, dtype=np.float64)
        sinr = np.zeros(n_users, dtype=np.float64)
        rates = np.zeros(n_users, dtype=np.float64)
        soft_qos = np.zeros(n_users, dtype=np.float64)
        binary_qos = np.zeros(n_users, dtype=np.int64)
        cluster_size = np.zeros(n_users, dtype=np.float64)
        min_ap_distance = np.zeros(n_users, dtype=np.float64)
        avg_cluster_beta = np.zeros(n_users, dtype=np.float64)
        primary_ap_load_norm = np.zeros(n_users, dtype=np.float64)

        for ue_idx in range(n_users):
            cluster = serving_clusters.get(ue_idx, [])
            cluster_size[ue_idx] = float(len(cluster))
            if distances_all[ue_idx].size > 0:
                min_ap_distance[ue_idx] = float(np.min(distances_all[ue_idx]))

            if not cluster or ue_idx not in active_users:
                continue

            ue_cache_key = tuple(np.round(np.array(ue_positions[ue_idx], dtype=np.float64), 4))
            h_useful = scenario.sample_small_scale_power(
                len(cluster),
                cache_key=("useful", ue_cache_key, ue_idx, tuple(cluster)),
            )
            for i, ap_idx in enumerate(cluster):
                p_lk = float(per_ue_power_map[ue_idx].get(ap_idx, 0.0))
                signal[ue_idx] += p_lk * float(beta_all[ue_idx][ap_idx]) * float(h_useful[i])

            avg_cluster_beta[ue_idx] = float(np.mean([beta_all[ue_idx][ap] for ap in cluster]))

            # Interference from streams intended to other users.
            for other_idx in range(n_users):
                if other_idx == ue_idx or other_idx not in active_users:
                    continue
                other_cluster = serving_clusters.get(other_idx, [])
                if not other_cluster:
                    continue
                h_int = scenario.sample_small_scale_power(
                    len(other_cluster),
                    cache_key=("interference", ue_cache_key, ue_idx, other_idx, tuple(other_cluster)),
                )
                beta_target = beta_all[ue_idx]
                for j, ap_idx in enumerate(other_cluster):
                    p_lj = float(per_ue_power_map[other_idx].get(ap_idx, 0.0))
                    interference[ue_idx] += p_lj * float(beta_target[ap_idx]) * float(h_int[j])

            interference[ue_idx] *= effective_interference_scale
            sinr[ue_idx] = signal[ue_idx] / max(interference[ue_idx] + noise_mw, 1e-20)
            rates[ue_idx] = (float(self.sce.BW) * np.log2(1.0 + sinr[ue_idx])) / 1e6
            soft_qos[ue_idx] = min(rates[ue_idx] / max(min_required_rate, 1e-6), 1.0)
            binary_qos[ue_idx] = int(rates[ue_idx] >= min_required_rate)
            primary_load = ap_user_count.get(int(primary_ap[ue_idx]), 0)
            max_load = max(1, max(ap_user_count.values(), default=1))
            primary_ap_load_norm[ue_idx] = float(primary_load) / float(max_load)

        aggregate_active_only = bool(getattr(self.sce, "qos_aggregate_active_only", True))
        if aggregate_active_only and np.any(active_mask):
            agg_soft_qos = soft_qos[active_mask]
            agg_binary_qos = binary_qos[active_mask]
            agg_rates = rates[active_mask]
        else:
            agg_soft_qos = soft_qos
            agg_binary_qos = binary_qos
            agg_rates = rates

        jain = self._jain_fairness(agg_rates)
        soft_qos_mean = float(np.mean(agg_soft_qos)) if agg_soft_qos.size > 0 else 0.0
        min_soft_qos = float(np.min(agg_soft_qos)) if agg_soft_qos.size > 0 else 0.0
        avg_rate = float(np.mean(agg_rates)) if agg_rates.size > 0 else 0.0
        total_power_mw = float(sum(sum(v.values()) for v in per_ue_power_map.values()))
        max_ap_mw = float(allocator.dbm_to_mw(getattr(scenario.cfg, "ap_max_tx_power_dbm", 23.0)))
        total_power_norm = total_power_mw / max(1.0, float(n_ap) * max_ap_mw)
        qos_violation_ratio = float(np.mean(1.0 - agg_binary_qos)) if agg_binary_qos.size > 0 else 1.0

        reward_vec = self._compose_reward(
            soft_qos,
            soft_qos_mean=soft_qos_mean,
            min_soft_qos=min_soft_qos,
            jain=jain,
            avg_rate=avg_rate,
            min_required_rate=min_required_rate,
            total_power_norm=total_power_norm,
            qos_violation_ratio=qos_violation_ratio,
            active_mask=active_mask,
        )

        cpu_loads = self._estimate_cpu_loads(scenario, ap_user_count)
        if hasattr(scenario, "set_visual_diagnostics"):
            scenario.set_visual_diagnostics(
                serving_clusters=serving_clusters,
                ap_loads=ap_user_count,
                cpu_loads=cpu_loads,
                step_metrics={
                    "rates_mbps": rates.tolist(),
                    "soft_qos": soft_qos.tolist(),
                    "binary_qos": binary_qos.tolist(),
                    "sinr_db": (10.0 * np.log10(np.maximum(sinr, 1e-12))).tolist(),
                    "signal_mw": signal.tolist(),
                    "interference_mw": interference.tolist(),
                    "cluster_size": cluster_size.tolist(),
                    "min_ap_distance_m": min_ap_distance.tolist(),
                    "avg_cluster_beta": avg_cluster_beta.tolist(),
                    "primary_ap_load_norm": primary_ap_load_norm.tolist(),
                    "jain_fairness": jain,
                    "mean_soft_qos": soft_qos_mean,
                    "min_soft_qos": min_soft_qos,
                    "mean_rate_mbps": avg_rate,
                    "min_rate_mbps": float(np.min(rates)) if n_users > 0 else 0.0,
                    "max_rate_mbps": float(np.max(rates)) if n_users > 0 else 0.0,
                    "rate_p5_mbps": float(np.percentile(rates, 5.0)) if n_users > 0 else 0.0,
                    "rate_p95_mbps": float(np.percentile(rates, 95.0)) if n_users > 0 else 0.0,
                    "binary_qos_rate": float(np.mean(binary_qos)) if n_users > 0 else 0.0,
                    "binary_qos_rate_active": float(np.mean(binary_qos[active_mask])) if np.any(active_mask) else 0.0,
                    "interference_mean_mw": float(np.mean(interference)) if n_users > 0 else 0.0,
                    "ap_load_mean": float(np.mean(list(ap_user_count.values()))) if ap_user_count else 0.0,
                    "ap_load_max": float(np.max(list(ap_user_count.values()))) if ap_user_count else 0.0,
                    "active_users": len(active_users),
                    "total_users": n_users,
                    "mean_soft_qos_all": float(np.mean(soft_qos)) if n_users > 0 else 0.0,
                    "mean_soft_qos_active": float(np.mean(soft_qos[active_mask])) if np.any(active_mask) else 0.0,
                    "min_rate_mbps_active": float(np.min(rates[active_mask])) if np.any(active_mask) else 0.0,
                    "rate_p5_mbps_active": float(np.percentile(rates[active_mask], 5.0)) if np.any(active_mask) else 0.0,
                    "interference_scale": effective_interference_scale,
                    "min_required_rate_mbps": float(min_required_rate),
                },
            )

        self._last_clusters = serving_clusters
        return {
            "rates_mbps": rates,
            "rewards": reward_vec,
            "binary_qos": binary_qos,
        }

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

    def _compute_ap_distances(self, aps, ue_pos_np: np.ndarray) -> np.ndarray:
        ap_pos = np.array([ap.get_location() for ap in aps], dtype=np.float64)
        return np.linalg.norm(ap_pos - ue_pos_np[None, :], axis=1)

    def _sample_active_users(self, n_users: int) -> Set[int]:
        if n_users <= 0:
            return set()
        activity_prob = float(getattr(self.sce, "user_activity_probability", 1.0) or 1.0)
        activity_prob = float(np.clip(activity_prob, 0.05, 1.0))
        active = [i for i in range(n_users) if np.random.rand() <= activity_prob]
        if not active:
            active = [int(np.random.randint(0, n_users))]

        max_active = int(getattr(self.sce, "max_active_users", 0) or 0)
        if max_active > 0 and len(active) > max_active:
            np.random.shuffle(active)
            active = active[:max_active]
        return set(active)

    def _required_rate_mbps(self, scenario) -> float:
        sinr_thr_linear = 10 ** (float(self.sce.QoS_thr) / 10.0)
        baseline = float(
            getattr(self.sce, "min_required_rate_mbps", None)
            or (float(self.sce.BW) * np.log2(1.0 + sinr_thr_linear)) / 1e6
        )
        factor = float(scenario.get_effective_min_rate_factor()) if hasattr(scenario, "get_effective_min_rate_factor") else 1.0
        return max(1e-6, baseline * max(0.1, factor))

    def _compose_reward(
        self,
        soft_qos: np.ndarray,
        soft_qos_mean: float,
        min_soft_qos: float,
        jain: float,
        avg_rate: float,
        min_required_rate: float,
        total_power_norm: float,
        qos_violation_ratio: float,
        active_mask: np.ndarray,
    ) -> np.ndarray:
        alpha = float(getattr(self.sce, "reward_alpha_soft_qos", 0.45) or 0.45)
        beta = float(getattr(self.sce, "reward_beta_min_soft_qos", 0.25) or 0.25)
        gamma = float(getattr(self.sce, "reward_gamma_fairness", 0.2) or 0.2)
        delta = float(getattr(self.sce, "reward_delta_avg_rate", 0.15) or 0.15)
        eta = float(getattr(self.sce, "reward_eta_power", 0.08) or 0.08)
        lambda_qos = float(getattr(self.sce, "reward_lambda_qos_violation", 0.25) or 0.25)
        indiv = float(getattr(self.sce, "reward_individual_soft_qos", 0.2) or 0.2)

        avg_rate_norm = avg_rate / max(min_required_rate, 1e-6)
        global_term = (
            alpha * soft_qos_mean
            + beta * min_soft_qos
            + gamma * jain
            + delta * avg_rate_norm
            - eta * total_power_norm
            - lambda_qos * qos_violation_ratio
        )
        indiv_active_only = bool(getattr(self.sce, "reward_individual_active_only", True))
        if indiv_active_only and active_mask.size == soft_qos.size:
            individual_term = soft_qos * active_mask.astype(np.float64)
        else:
            individual_term = soft_qos

        reward = global_term + indiv * individual_term

        norm_scale = float(getattr(self.sce, "reward_norm_scale", 1.0) or 1.0)
        clip_abs = float(getattr(self.sce, "reward_clip_abs", 3.0) or 3.0)
        reward = reward / max(1e-6, norm_scale)
        reward = np.clip(reward, -clip_abs, clip_abs)
        return reward.astype(np.float64)

    def _jain_fairness(self, rates: np.ndarray) -> float:
        if rates.size <= 0:
            return 0.0
        s = float(np.sum(rates))
        s2 = float(np.sum(np.square(rates)))
        if s2 <= 1e-20:
            return 0.0
        return (s * s) / (float(rates.size) * s2)

    def _infer_ue_index(self, agent_location: torch.Tensor, scenario) -> int:
        ue_positions = scenario.get_current_ue_positions() or []
        if not ue_positions:
            return -1
        p = self._to_numpy(agent_location).astype(np.float64)
        d = [float(np.linalg.norm(np.array(q, dtype=np.float64) - p)) for q in ue_positions]
        if not d:
            return -1
        return int(np.argmin(np.array(d, dtype=np.float64)))

    def _compute_useful_signal(self, *args, **kwargs) -> float:
        return 0.0

    def _compute_interference(self, *args, **kwargs) -> float:
        return 0.0

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
