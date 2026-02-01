"""Network metrics real-time plotting utilities.

Provides a plotting class for visualizing DQN training metrics like loss,
Q-values, gradients, and TD errors. Updates a single PNG file that can be
monitored during training.

Design goals:
  - Non-intrusive: if matplotlib is missing, plotting is silently disabled.
  - Throttled: only redraw every `plot_interval` steps to avoid overhead.
  - Headless-friendly: uses Agg backend.
  - Separate from environment metrics for clarity.

Usage:
    from functions.network_metrics_plot import NetworkMetricsPlotter
    plotter = NetworkMetricsPlotter(enabled=True, plot_interval=50)
    plotter.update(
        step=global_step,
        loss=0.05,
        mean_q=2.3,
        max_q=5.1,
        min_q=-0.2,
        q_std=1.2,
        grad_norm=0.8,
        target_q_mean=2.1,
        td_error=0.15,
    )
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

from constants import (
    NETWORK_PLOT_ENABLED,
    NETWORK_PLOT_INTERVAL,
    NETWORK_PLOT_SMOOTH_WINDOW,
    NETWORK_PLOT_FILENAME,
    NETWORK_PLOT_FIGSIZE,
    NETWORK_PLOT_DPI,
)

# Lazy import matplotlib; handle absence gracefully
try:
    import matplotlib

    matplotlib.use("Agg")  # Force non-interactive backend
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - if matplotlib not present
    plt = None  # type: ignore


class NetworkMetricsPlotter:
    """Real-time plotter for DQN network training metrics."""

    def __init__(
        self,
        enabled: bool = NETWORK_PLOT_ENABLED,
        plot_interval: int = NETWORK_PLOT_INTERVAL,
        out_path: str = NETWORK_PLOT_FILENAME,
        smooth_window: int = NETWORK_PLOT_SMOOTH_WINDOW,
        x_axis_mode: str = "steps",
    ):
        """Initialize the network metrics plotter.

        Args:
            enabled: Whether plotting is enabled.
            plot_interval: Number of steps between plot updates.
            out_path: Path to save the plot image.
            smooth_window: Window size for moving average smoothing.
            x_axis_mode: 'steps' or 'episodes' for x-axis labeling.
        """
        self.enabled = enabled and plt is not None
        self.plot_interval = max(1, int(plot_interval))
        self.out_path = out_path
        self.smooth_window = max(1, int(smooth_window))
        self.x_axis_mode = (
            x_axis_mode if x_axis_mode in ("steps", "episodes") else "steps"
        )

        self.history: Dict[str, List[float]] = {
            "step": [],
            "loss": [],
            "mean_q": [],
            "max_q": [],
            "min_q": [],
            "q_std": [],
            "grad_norm": [],
            "target_q_mean": [],
            "td_error": [],
        }

        self._has_data = False

        if self.enabled:
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)

    def update(
        self,
        step: int,
        loss: Optional[float] = None,
        mean_q: Optional[float] = None,
        max_q: Optional[float] = None,
        min_q: Optional[float] = None,
        q_std: Optional[float] = None,
        grad_norm: Optional[float] = None,
        target_q_mean: Optional[float] = None,
        td_error: Optional[float] = None,
    ):
        """Update metrics history and optionally render plot.

        Args:
            step: Current global step.
            loss: Training loss value.
            mean_q: Mean Q-value from policy network.
            max_q: Maximum Q-value.
            min_q: Minimum Q-value.
            q_std: Standard deviation of Q-values.
            grad_norm: Gradient norm.
            target_q_mean: Mean target Q-value.
            td_error: Mean TD error.
        """
        if not self.enabled:
            return

        if all(v is None for v in [loss, mean_q, max_q, min_q, q_std, grad_norm, target_q_mean, td_error]):
            return

        self._has_data = True
        self.history["step"].append(step)

        def _append(key: str, value: Optional[float]):
            if value is not None:
                self.history[key].append(value)
            else:
                last = self.history[key][-1] if self.history[key] else 0.0
                self.history[key].append(last)

        _append("loss", loss)
        _append("mean_q", mean_q)
        _append("max_q", max_q)
        _append("min_q", min_q)
        _append("q_std", q_std)
        _append("grad_norm", grad_norm)
        _append("target_q_mean", target_q_mean)
        _append("td_error", td_error)

        if step % self.plot_interval != 0 and step > 0:
            return
        self._render()

    def _moving_average(self, data: List[float]) -> List[float]:
        """Compute moving average with window size."""
        if len(data) < 2:
            return data
        w = self.smooth_window
        out: List[float] = []
        sum = 0.0
        for i, v in enumerate(data):
            sum += v
            if i >= w:
                sum -= data[i - w]
            out.append(sum / min(i + 1, w))
        return out

    def _render(self):  
        """Render the metrics plot to file."""
        if plt is None or not self._has_data:
            return

        step = self.history["step"]
        if not step:
            return

        loss = self.history["loss"]
        mean_q = self.history["mean_q"]
        max_q = self.history["max_q"]
        min_q = self.history["min_q"]
        q_std = self.history["q_std"]
        grad_norm = self.history["grad_norm"]
        target_q_mean = self.history["target_q_mean"]
        td_error = self.history["td_error"]

        loss_ma = self._moving_average(loss)
        td_error_ma = self._moving_average(td_error)
        grad_norm_ma = self._moving_average(grad_norm)

        plt.close("all")
        fig, axes = plt.subplots(2, 3, figsize=NETWORK_PLOT_FIGSIZE)
        x_label = "Episode" if self.x_axis_mode == "episodes" else "Step"

        ax = axes[0, 0]
        ax.plot(step, loss, alpha=0.3, label="loss", color="tab:red")
        ax.plot(
            step,
            loss_ma,
            label=f"loss MA (w={self.smooth_window})",
            color="darkred",
        )
        ax.set_title("Training Loss")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.plot(step, mean_q, label="mean Q", color="tab:blue")
        ax.plot(step, max_q, label="max Q", color="tab:green", alpha=0.7)
        ax.plot(step, min_q, label="min Q", color="tab:orange", alpha=0.7)
        ax.fill_between(step, min_q, max_q, alpha=0.2, color="tab:blue")
        ax.set_title("Q-Value Statistics")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Q-Value")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

        ax = axes[0, 2]
        ax.plot(step, td_error, alpha=0.3, label="TD error", color="tab:purple")
        ax.plot(
            step,
            td_error_ma,
            label=f"TD error MA (w={self.smooth_window})",
            color="purple",
        )
        ax.set_title("TD Error")
        ax.set_xlabel(x_label)
        ax.set_ylabel("TD Error")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(step, grad_norm, alpha=0.3, label="grad norm", color="tab:cyan")
        ax.plot(
            step,
            grad_norm_ma,
            label=f"grad norm MA (w={self.smooth_window})",
            color="darkcyan",
        )
        ax.set_title("Gradient Norm")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Norm")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(step, q_std, label="Q std", color="tab:olive")
        ax.set_title("Q-Value Standard Deviation")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Std Dev")
        ax.grid(alpha=0.3)

        ax = axes[1, 2]
        ax.plot(step, mean_q, label="policy Q mean", color="tab:blue")
        ax.plot(step, target_q_mean, label="target Q mean", color="tab:red", linestyle="--")
        ax.set_title("Policy vs Target Q-Values")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean Q-Value")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

        fig.suptitle("Network Training Metrics", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(self.out_path, dpi=NETWORK_PLOT_DPI)

    def close(self):
        """Clean up resources (placeholder for future use)."""
        pass


__all__ = ["NetworkMetricsPlotter"]
