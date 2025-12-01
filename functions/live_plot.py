"""Real-time training plot utilities."""

from __future__ import annotations

import os
from typing import List, Dict, Optional

# Lazy import matplotlib; handle absence gracefully
try:
    import matplotlib

    matplotlib.use("Agg")  # Force non-interactive backend
    import matplotlib.pyplot as plt
except Exception:
    plt = None


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
            "return": [],  # Added return tracking
            "qos": [],
            "capacity_sum_mbps": [],
        }

        if self.enabled:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    # Updated signature to accept optional episode_return
    def update(
        self,
        step: int,
        epsilon: float,
        mean_reward: float,
        qos: float,
        capacity_sum_mbps: float | None = None,
        episode_return: float | None = None,
    ):
        if not self.enabled:
            return
        self.history["step"].append(step)
        self.history["epsilon"].append(epsilon)
        self.history["mean_reward"].append(mean_reward)
        self.history["qos"].append(qos)

        # Handle optional metrics and maintain alignment
        def append_metric(key, value):
            if value is not None:
                self.history[key].append(value)
            else:
                # If value is missing (e.g., return when plotting by step), repeat last or use 0
                last_val = self.history[key][-1] if self.history[key] else 0.0
                self.history[key].append(last_val)

        append_metric("capacity_sum_mbps", capacity_sum_mbps)
        append_metric("return", episode_return)

        # Throttle rendering
        if step % self.plot_interval != 0 and step > 0:
            return
        self._render()

    # --- internal helpers ---
    def _moving_average(self, data: List[float]) -> List[float]:
        # Simple Moving Average implementation that handles the ramp-up phase
        if len(data) < 2:
            return data
        w = self.smooth_window
        out: List[float] = []
        cumsum = 0.0
        for i, v in enumerate(data):
            cumsum += v
            if i >= w:
                cumsum -= data[i - w]
            # Use available data size during ramp-up
            out.append(cumsum / min(i + 1, w))
        return out

    def _render(self):
        if plt is None:
            return

        # Prepare data
        step = self.history["step"]
        if not step:
            return

        eps = self.history["epsilon"]
        rew = self.history["mean_reward"]
        ret = self.history["return"]
        qos = self.history["qos"]
        cap = self.history["capacity_sum_mbps"]

        # Calculate moving averages
        rew_ma = self._moving_average(rew)
        cap_ma = self._moving_average(cap)
        ret_ma = self._moving_average(ret)
        qos_ma = self._moving_average(qos)

        plt.close("all")
        # Increased figure size for better visualization (2x3 grid)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        x_label = "Episode" if self.x_axis_mode == "episodes" else "Step"

        # Plot 1: Average Reward
        ax = axes[0][0]
        ax.plot(step, rew, alpha=0.3, label="Avg Reward")
        ax.plot(
            step,
            rew_ma,
            label=f"Avg Reward MA (w={self.smooth_window})",
            color="tab:green",
        )
        ax.set_title("Average Reward")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: QoS Satisfaction
        ax = axes[0][1]
        ax.plot(step, qos, alpha=0.3, color="tab:purple", label="QoS Rate")
        ax.plot(
            step, qos_ma, color="tab:purple", label=f"QoS MA (w={self.smooth_window})"
        )
        ax.set_title("QoS Satisfaction (Mean)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("QoS Rate")
        ax.set_ylim(-0.05, 1.05)  # QoS rate is bounded [0, 1]
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 3: Return (New Plot)
        ax = axes[0][2]
        ax.plot(step, ret, alpha=0.3, label="Return")
        ax.plot(
            step, ret_ma, label=f"Return MA (w={self.smooth_window})", color="tab:red"
        )
        ax.set_title("Episode Return (Cumulative Reward)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 4: Epsilon Schedule
        ax = axes[1][0]
        ax.plot(step, eps, color="tab:orange")
        ax.set_title("Epsilon Schedule (Exploration Rate)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Epsilon")
        ax.grid(alpha=0.3)

        # Plot 5: System Capacity
        ax = axes[1][1]
        ax.plot(step, cap, alpha=0.3, label="Capacity Sum (Mbps)", color="tab:cyan")
        ax.plot(
            step,
            cap_ma,
            label=f"Capacity MA (w={self.smooth_window})",
            color="tab:blue",
        )
        ax.set_title("System Capacity (Mbps)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Capacity (Mbps)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 6: Reward Scatter
        ax = axes[1][2]
        ax.scatter(step, rew, s=8, alpha=0.4)
        ax.set_title("Reward Scatter")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Reward")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        try:
            fig.savefig(self.out_path, dpi=120)
        except Exception as e:
            print(f"Warning: Failed to save plot: {e}")

    def close(self):
        pass


__all__ = ["RealTimeStepPlotter"]
