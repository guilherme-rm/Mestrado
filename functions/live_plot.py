"""Real-time training plot utilities.

Moved from project root to `functions` package for modularity.
"""
from __future__ import annotations

import os
from typing import List, Dict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore


class RealTimeStepPlotter:
    def __init__(self, enabled: bool = True, plot_interval: int = 50, out_path: str = "Result/training_progress.png", smooth_window: int = 100, x_axis_mode: str = "steps"):
        self.enabled = enabled and plt is not None
        self.plot_interval = max(1, int(plot_interval))
        self.out_path = out_path
        self.smooth_window = max(1, int(smooth_window))
        self.x_axis_mode = x_axis_mode if x_axis_mode in ("steps", "episodes") else "steps"
        self.history: Dict[str, List[float]] = {
            "step": [],
            "epsilon": [],
            "mean_reward": [],
            "qos": [],
            "capacity_sum_mbps": [],
        }
        if self.enabled:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def update(self, step: int, epsilon: float, mean_reward: float, qos: float, capacity_sum_mbps: float | None = None):
        if not self.enabled:
            return
        self.history["step"].append(step)
        self.history["epsilon"].append(epsilon)
        self.history["mean_reward"].append(mean_reward)
        self.history["qos"].append(qos)
        if capacity_sum_mbps is not None:
            self.history["capacity_sum_mbps"].append(capacity_sum_mbps)
        else:
            last = self.history["capacity_sum_mbps"][-1] if self.history["capacity_sum_mbps"] else 0.0
            self.history["capacity_sum_mbps"].append(last)
        if step % self.plot_interval != 0 and step > 0:
            return
        self._render()

    def _moving_average(self, data: List[float]) -> List[float]:
        if len(data) < 2:
            return data
        w = self.smooth_window
        out: List[float] = []
        cumsum = 0.0
        for i, v in enumerate(data):
            cumsum += v
            if i >= w:
                cumsum -= data[i - w]
            out.append(cumsum / min(i + 1, w))
        return out

    def _render(self):  # pragma: no cover
        if plt is None:
            return
        step = self.history["step"]
        eps = self.history["epsilon"]
        rew = self.history["mean_reward"]
        qos = self.history["qos"]
        cap = self.history["capacity_sum_mbps"]
        rew_ma = self._moving_average(rew)
        cap_ma = self._moving_average(cap)
        plt.close("all")
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        x_label = "Episode" if self.x_axis_mode == "episodes" else "Step"
        ax = axes[0][0]
        ax.plot(step, rew, alpha=0.3, label="reward")
        ax.plot(step, rew_ma, label=f"reward MA (w={self.smooth_window})", color="tab:green")
        ax.set_title("Mean Reward per Step")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(alpha=0.3)
        ax = axes[0][1]
        ax.plot(step, qos, color="tab:purple")
        ax.set_title("QoS Satisfaction (Mean)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("QoS Rate")
        ax.grid(alpha=0.3)
        ax = axes[1][0]
        ax.plot(step, eps, color="tab:orange")
        ax.set_title("Epsilon Schedule")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Epsilon")
        ax.grid(alpha=0.3)
        ax = axes[1][1]
        ax.plot(step, cap, alpha=0.3, label="capacity sum (Mbps)", color="tab:cyan")
        ax.plot(step, cap_ma, label=f"capacity MA (w={self.smooth_window})", color="tab:blue")
        ax.set_title("System Capacity (Mbps)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Capacity (Mbps)")
        ax.legend()
        ax.grid(alpha=0.3)
        ax = axes[1][2]
        if step:
            ax.scatter(step, rew, s=8, alpha=0.4)
        ax.set_title("Reward Scatter")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Reward")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.out_path, dpi=120)

    def close(self):
        pass


__all__ = ["RealTimeStepPlotter"]