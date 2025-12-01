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
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    if isinstance(obj, dict):
        return dict(obj)
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
    # Updated HEADER to include 'return_val'
    HEADER = [
        "episode",
        "steps",
        "return_val",
        "avg_reward",
        "qos_rate",
        "capacity_mean",
        "epsilon_last",
        "duration_seconds",
    ]

    def __init__(self, run_dir: RunDirectoryManager):
        self.logger = CSVLogger(run_dir.subpath("episode_metrics.csv"), self.HEADER)

    def log(self, **metrics):
        row = [metrics.get(k, "") for k in self.HEADER]
        self.logger.log(row)

    def close(self):
        self.logger.close()


class StepMetricsLogger:
    # (Implementation remains the same)
    HEADER = [
        "global_step",
        "episode",
        "step_in_episode",
        "epsilon",
        "mean_reward",
        "qos_mean",
        "capacity_sum_mbps",
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


__all__ = [
    "RunDirectoryManager",
    "save_config",
    "save_environment_snapshot",
    "EpisodeMetricsLogger",
    "StepMetricsLogger",
    "checkpoint_agents",
    "write_summary",
]
