"""Logging and run directory management utilities."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable, List, Optional

import torch


def _serialize_config(obj) -> Dict[str, Any]:
    # Helper to serialize configuration objects (like DotDic)
    # Check for dict first - DotDic is a dict subclass, so this catches it
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


class RunDirectoryManager:
    def __init__(
        self, root: str = "Result", prefix: str = "run", overwrite: bool = False
    ):
        if overwrite:
            self.path = Path(root) / prefix
            if self.path.exists():
                try:
                    shutil.rmtree(self.path)
                except OSError as e:
                    print(f"Warning: Could not overwrite {self.path}: {e}")
            self.path.mkdir(parents=True, exist_ok=True)
        else:
            ts = time.strftime("%Y%m%d-%H%M%S")
            self.path = Path(root) / f"{prefix}_{ts}"
            self.path.mkdir(parents=True, exist_ok=True)

    def subpath(self, name: str) -> Path:
        p = self.path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def save_config(run_dir: RunDirectoryManager, opt, sce):
    (run_dir.subpath("opt.json")).write_text(
        json.dumps(_serialize_config(opt), indent=2)
    )
    (run_dir.subpath("sce.json")).write_text(
        json.dumps(_serialize_config(sce), indent=2)
    )


def save_environment_snapshot(run_dir: RunDirectoryManager):
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    (run_dir.subpath("environment.json")).write_text(json.dumps(env, indent=2))


class CSVLogger:
    def __init__(self, path: Path, header: List[str]):
        self.path = path
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.f)
        self.writer.writerow(header)
        self.f.flush()

    def log(self, row: Iterable[Any]):
        self.writer.writerow(list(row))
        self.f.flush()

    def close(self):
        if not self.f.closed:
            self.f.close()


class EpisodeMetricsLogger:
    """Logs episode-level metrics for training analysis and diagnosis.
    
    Diagnostic metrics included:
    - reward_min/max/std: Reward distribution analysis per episode
    - loss_avg: Average training loss (model learning rate)
    - mean_q_avg: Average Q-value estimates (value drift detection)
    - td_error_avg: TD error magnitude (learning signal quality)
    - grad_norm_avg: Gradient norm (vanishing/exploding gradient detection)
    - learning_steps: Number of actual gradient updates (learning activity)
    - memory_size: Replay buffer fill level (experience accumulation)
    - success_rate: Fraction of steps meeting QoS threshold
    """
    HEADER = [
        "episode",
        "steps",
        "return_val",
        "avg_reward",
        "reward_min",
        "reward_max",
        "reward_std",
        "qos_rate",
        "capacity_mean",
        "epsilon_last",
        "duration_seconds",
        # Learning diagnostics
        "loss_avg",
        "mean_q_avg",
        "max_q_avg",
        "td_error_avg",
        "grad_norm_avg",
        "learning_steps",
        "memory_size",
        "success_rate",
    ]

    def __init__(self, run_dir: RunDirectoryManager):
        self.logger = CSVLogger(run_dir.subpath("episode_metrics.csv"), self.HEADER)

    def log(self, **metrics):
        row = [metrics.get(k, "") for k in self.HEADER]
        self.logger.log(row)

    def close(self):
        self.logger.close()


class StepMetricsLogger:
    """Logs step-level metrics for detailed training diagnosis.
    
    Diagnostic metrics included:
    - loss: Training loss (should decrease then stabilize)
    - mean_q: Average Q-values (monitor for drift/explosion)
    - max_q: Maximum Q-value (detect divergence)
    - min_q: Minimum Q-value (detect collapse)
    - q_std: Q-value standard deviation (action differentiation)
    - td_error: Temporal difference error (learning signal strength)
    - grad_norm: Gradient norm (detect vanishing/exploding gradients)
    - target_q_mean: Target network Q-values (target stability)
    - memory_size: Replay buffer fill level
    - did_learn: Whether learning occurred this step
    """
    HEADER = [
        "global_step",
        "episode",
        "step_in_episode",
        "epsilon",
        "mean_reward",
        "qos_mean",
        "capacity_sum_mbps",
        # Learning diagnostics
        "loss",
        "mean_q",
        "max_q",
        "min_q",
        "q_std",
        "td_error",
        "grad_norm",
        "target_q_mean",
        "memory_size",
        "did_learn",
    ]

    def __init__(self, run_dir: RunDirectoryManager, throttle: Optional[int] = 1):
        try:
            t = int(throttle) if throttle is not None else 1
        except (TypeError, ValueError):
            t = 1
        self.throttle = max(1, t)
        self.logger = CSVLogger(run_dir.subpath("step_metrics.csv"), self.HEADER)

    def log(self, **metrics):
        gs = metrics.get("global_step", 0)
        if gs % self.throttle != 0:
            return
        row = [metrics.get(k, "") for k in self.HEADER]
        self.logger.log(row)

    def close(self):
        self.logger.close()


def checkpoint_agents(
    run_dir: RunDirectoryManager, agents, episode: int, tag: str = ""
):
    ckpt_dir = run_dir.subpath("checkpoints")
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    for i, ag in enumerate(agents):
        fp = ckpt_dir / f"agent{i}_ep{episode}{'_'+tag if tag else ''}.pt"
        # Save model and optimizer state for potential resume
        torch.save(
            {
                "episode": episode,
                "model_state_dict": ag.model_policy.state_dict(),
                "optimizer_state_dict": ag.optimizer.state_dict(),
            },
            fp,
        )


def write_summary(run_dir: RunDirectoryManager, metrics: Dict[str, Any]):
    (run_dir.subpath("summary.json")).write_text(json.dumps(metrics, indent=2))


def write_experiment_summary(
    run_dir: RunDirectoryManager,
    opt,
    sce,
    total_episodes: int,
    total_steps: int,
    total_duration_seconds: float,
    final_metrics: Dict[str, Any],
    best_episode: Optional[Dict[str, Any]] = None,
    learning_diagnostics: Optional[Dict[str, Any]] = None,
    convergence_info: Optional[Dict[str, Any]] = None,
    feature_flags: Optional[Dict[str, Any]] = None,
    resource_usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate and save a comprehensive experiment summary for article comparison.
    
    This summary provides all relevant information about an experiment run,
    suitable for comparing different models and configurations in publications.
    
    Args:
        run_dir: The run directory manager
        opt: Training options (DotDic or dict)
        sce: Scenario configuration (DotDic or dict)
        total_episodes: Number of episodes completed
        total_steps: Total environment steps taken
        total_duration_seconds: Wall-clock training time
        final_metrics: Summary of final performance metrics
        best_episode: Information about the best performing episode
        learning_diagnostics: Learning-related metrics (loss, Q-values, etc.)
        convergence_info: Convergence detection information
        feature_flags: Dictionary of feature flags (GNN, mobility, etc.)
        resource_usage: Resource utilization summary (CPU/GPU usage)
    
    Returns:
        The complete summary dictionary
    """
    # Serialize configs
    opt_dict = _serialize_config(opt)
    sce_dict = _serialize_config(sce)
    
    # Environment snapshot
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    
    # Build comprehensive summary
    summary = {
        # ===== Experiment Metadata =====
        "experiment": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_directory": str(run_dir.path),
            "total_episodes": total_episodes,
            "total_steps": total_steps,
            "wall_time_seconds": round(total_duration_seconds, 2),
            "wall_time_minutes": round(total_duration_seconds / 60, 2),
            "steps_per_second": round(total_steps / total_duration_seconds, 2) if total_duration_seconds > 0 else 0,
        },
        
        # ===== Scenario Configuration =====
        "scenario": {
            "base_stations": {
                "macro": sce_dict.get("nMBS", 0),
                "pico": sce_dict.get("nPBS", 0),
                "femto": sce_dict.get("nFBS", 0),
                "total": sce_dict.get("nMBS", 0) + sce_dict.get("nPBS", 0) + sce_dict.get("nFBS", 0),
            },
            "coverage_radii": {
                "macro_m": sce_dict.get("rMBS"),
                "pico_m": sce_dict.get("rPBS"),
                "femto_m": sce_dict.get("rFBS"),
            },
            "radio": {
                "bandwidth_hz": sce_dict.get("BW"),
                "num_channels": sce_dict.get("nChannel"),
                "noise_floor_dbm_hz": sce_dict.get("N0"),
            },
            "qos_threshold_mbps": sce_dict.get("QoS_thr"),
            "reward_shaping": {
                "profit": sce_dict.get("profit"),
                "power_cost": sce_dict.get("power_cost"),
                "action_cost": sce_dict.get("action_cost"),
                "negative_cost": sce_dict.get("negative_cost"),
            },
        },
        
        # ===== Training Configuration =====
        "training": {
            "num_agents": opt_dict.get("nagents"),
            "episodes": opt_dict.get("nepisodes"),
            "max_steps_per_episode": opt_dict.get("nsteps"),
            "early_termination": opt_dict.get("early_termination", True),
            "hyperparameters": {
                "learning_rate": opt_dict.get("learning_rate"),
                "gamma": opt_dict.get("gamma"),
                "tau": opt_dict.get("tau"),
                "batch_size": opt_dict.get("batch_size"),
                "replay_buffer_capacity": opt_dict.get("capacity"),
                "min_memory_for_learning": opt_dict.get("min_memory_for_learning"),
            },
            "exploration": {
                "policy": opt_dict.get("epsilon_policy"),
                "action_strategy": opt_dict.get("action_strategy"),
                "eps_min": opt_dict.get("eps_min"),
                "eps_max": opt_dict.get("eps_max"),
                "eps_increment": opt_dict.get("eps_increment"),
                "eps_start": opt_dict.get("eps_start"),
                "eps_end": opt_dict.get("eps_end"),
                "eps_decay_steps": opt_dict.get("eps_decay_steps"),
            },
        },
        
        # ===== Feature Flags =====
        "features": feature_flags or {},
        
        # ===== Environment =====
        "environment": env_info,
        
        # ===== Performance Metrics =====
        "performance": {
            "final": final_metrics,
            "best_episode": best_episode,
            "convergence": convergence_info,
        },
        
        # ===== Learning Diagnostics =====
        "learning_diagnostics": learning_diagnostics or {},
        
        # ===== Resource Usage =====
        "resource_usage": resource_usage or {},
    }
    
    # Write to file
    (run_dir.subpath("experiment_summary.json")).write_text(
        json.dumps(summary, indent=2, default=str)
    )
    
    # Also write a compact version for quick reference
    compact_summary = _build_compact_summary(summary)
    (run_dir.subpath("summary.json")).write_text(
        json.dumps(compact_summary, indent=2, default=str)
    )
    
    return summary


def _build_compact_summary(full_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact summary with key metrics for quick reference."""
    exp = full_summary.get("experiment", {})
    sce = full_summary.get("scenario", {})
    train = full_summary.get("training", {})
    perf = full_summary.get("performance", {})
    features = full_summary.get("features", {})
    resources = full_summary.get("resource_usage", {})
    
    final = perf.get("final", {})
    best = perf.get("best_episode", {})
    
    return {
        # Quick identification
        "timestamp": exp.get("timestamp"),
        "wall_time_minutes": exp.get("wall_time_minutes"),
        
        # Key configuration
        "num_agents": train.get("num_agents"),
        "total_base_stations": sce.get("base_stations", {}).get("total"),
        "action_strategy": train.get("exploration", {}).get("action_strategy"),
        "gnn_enabled": features.get("gnn_enabled"),
        "mobility_enabled": features.get("mobility_enabled"),
        
        # Key performance metrics
        "final_avg_reward": final.get("mean_reward"),
        "final_avg_return": final.get("mean_return"),
        "final_qos_rate": final.get("mean_qos"),
        
        # Best episode
        "best_episode_num": best.get("episode") if best else None,
        "best_episode_return": best.get("return") if best else None,
        "best_episode_qos": best.get("qos_rate") if best else None,
        
        # Training stats
        "total_episodes": exp.get("total_episodes"),
        "total_steps": exp.get("total_steps"),
        "steps_per_second": exp.get("steps_per_second"),
        
        # Resource usage (key metrics only)
        "cpu_mean_percent": resources.get("cpu_mean"),
        "memory_max_percent": resources.get("memory_max"),
        "gpu_utilization_mean": resources.get("gpu_utilization_mean"),
    }


__all__ = [
    "RunDirectoryManager",
    "save_config",
    "save_environment_snapshot",
    "EpisodeMetricsLogger",
    "StepMetricsLogger",
    "checkpoint_agents",
    "write_summary",
    "write_experiment_summary",
]
