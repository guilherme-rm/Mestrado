"""Real-time training plot utilities.

Provides a lightweight plotting class that can be updated each training step.
It maintains in-memory metric history and re-renders a single PNG file
(`Result/training_progress.png`) overwriting it so disk does not fill.

Design goals:
  - Non-intrusive: if matplotlib is missing, plotting is silently disabled.
  - Throttled: only redraw every `plot_interval` steps to avoid overhead.
  - Headless-friendly: uses Agg backend.
  - Minimal dependencies: just matplotlib (optional) + torch for type hints.

Usage:
  from live_plot import RealTimeStepPlotter
  plotter = RealTimeStepPlotter(enabled=opt.enable_plot, plot_interval=opt.plot_interval)
  plotter.update(step=global_step, epsilon=eps, mean_reward=float(rewards.mean()), qos=float(qos.mean()))

Metrics captured:
  - step (global)
  - epsilon
  - mean_reward (per step mean, optionally smoothed when drawing)
  - qos (mean QoS satisfaction)

You can extend by adding loss, memory usage, etc. Provide them in update().
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

# Lazy import matplotlib; handle absence gracefully
try:
    import matplotlib

    matplotlib.use("Agg")  # Force non-interactive backend
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - if matplotlib not present
    plt = None  # type: ignore


class RealTimeStepPlotter:
    def __init__(
        self,
        enabled: bool = True,
        plot_interval: int = 50,
        out_path: str = "Result/training_progress.png",
        smooth_window: int = 100,
        x_axis_mode: str = "steps",
    ):
        self.enabled = enabled and plt is not None
        self.plot_interval = max(1, int(plot_interval))
        self.out_path = out_path
        self.smooth_window = max(1, int(smooth_window))
        self.x_axis_mode = (
            x_axis_mode if x_axis_mode in ("steps", "episodes") else "steps"
        )

        self.history: Dict[str, List[float]] = {
            "step": [],
            "epsilon": [],
            "mean_reward": [],
            "qos": [],
            "capacity_sum_mbps": [],  # system capacity (sum of per-agent capacities)
        }

        if self.enabled:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def update(
        self,
        step: int,
        epsilon: float,
        mean_reward: float,
        qos: float,
        capacity_sum_mbps: float | None = None,
    ):
        if not self.enabled:
            return
        self.history["step"].append(step)
        self.history["epsilon"].append(epsilon)
        self.history["mean_reward"].append(mean_reward)
        self.history["qos"].append(qos)
        if capacity_sum_mbps is not None:
            self.history["capacity_sum_mbps"].append(capacity_sum_mbps)
        else:
            # Maintain alignment for indexing; repeat last or 0
            last = (
                self.history["capacity_sum_mbps"][-1]
                if self.history["capacity_sum_mbps"]
                else 0.0
            )
            self.history["capacity_sum_mbps"].append(last)

        # For episodes mode we still throttle by plot_interval but 'step' is semantic x-value
        if step % self.plot_interval != 0 and step > 0:
            return
        self._render()

    # --- internal helpers ---
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

    def _render(self):  # pragma: no cover (simple plotting)
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
        ax.plot(
            step, rew_ma, label=f"reward MA (w={self.smooth_window})", color="tab:green"
        )
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
        ax.plot(
            step,
            cap_ma,
            label=f"capacity MA (w={self.smooth_window})",
            color="tab:blue",
        )
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
        # Nothing persistent yet, placeholder for future resources
        pass


__all__ = ["RealTimeStepPlotter"]
