"""Real-time training plot utilities."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class BasePlotter(ABC):
    """Base class for all training plotters."""

    def __init__(
        self,
        enabled: bool = True,
        plot_interval: int = 50,
        out_path: str = "Result/plot.png",
        smooth_window: int = 100,
    ):
        self.enabled = enabled and plt is not None
        self.plot_interval = max(1, int(plot_interval))
        self.out_path = out_path
        self.smooth_window = max(1, int(smooth_window))
        self._update_count = 0
        self.history: Dict[str, List[float]] = {}

        if self.enabled:
            out_dir = os.path.dirname(self.out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

    def _moving_average(self, data: List[float]) -> List[float]:
        """Simple Moving Average that handles the ramp-up phase."""
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

    def _should_render(self) -> bool:
        """Check if we should render based on update count and interval."""
        return self._update_count % self.plot_interval == 0

    def _safe_save(self, fig):
        """Safely save figure to file."""
        try:
            fig.savefig(self.out_path, dpi=150, bbox_inches="tight")
        except Exception as e:
            print(f"Warning: Failed to save plot to {self.out_path}: {e}")
        finally:
            plt.close(fig)

    @abstractmethod
    def _render(self):
        """Render the plot. Must be implemented by subclasses."""
        pass

    def finalize(self):
        """Force a final render."""
        if self.enabled and self.history.get("step"):
            self._render()

    def close(self):
        """Cleanup and finalize."""
        self.finalize()


class RealTimeStepPlotter(BasePlotter):
    """Plots training progress metrics (rewards, QoS, capacity, etc.)."""

    def __init__(
        self,
        enabled: bool = True,
        plot_interval: int = 50,
        out_path: str = "Result/training_progress.png",
        smooth_window: int = 100,
        x_axis_mode: str = "steps",
    ):
        super().__init__(enabled, plot_interval, out_path, smooth_window)
        self.x_axis_mode = x_axis_mode if x_axis_mode in ("steps", "episodes") else "steps"

        self.history: Dict[str, List[float]] = {
            "step": [],
            "epsilon": [],
            "mean_reward": [],
            "return": [],
            "qos": [],
            "capacity_sum_mbps": [],
        }

    def update(
        self,
        step: int,
        epsilon: float,
        mean_reward: float,
        qos: float,
        capacity_sum_mbps: Optional[float] = None,
        episode_return: Optional[float] = None,
    ):
        """Record metrics and periodically update the plot."""
        if not self.enabled:
            return

        self._update_count += 1
        self.history["step"].append(step)
        self.history["epsilon"].append(epsilon)
        self.history["mean_reward"].append(mean_reward)
        self.history["qos"].append(qos)

        def append_metric(key: str, value: Optional[float]):
            if value is not None:
                self.history[key].append(value)
            else:
                last_val = self.history[key][-1] if self.history[key] else 0.0
                self.history[key].append(last_val)

        append_metric("capacity_sum_mbps", capacity_sum_mbps)
        append_metric("return", episode_return)

        if self._should_render():
            self._render()

    def _render(self):
        if plt is None:
            return

        step = self.history["step"]
        if not step:
            return

        eps = self.history["epsilon"]
        rew = self.history["mean_reward"]
        ret = self.history["return"]
        qos = self.history["qos"]
        cap = self.history["capacity_sum_mbps"]

        rew_ma = self._moving_average(rew)
        cap_ma = self._moving_average(cap)
        ret_ma = self._moving_average(ret)
        qos_ma = self._moving_average(qos)

        plt.close("all")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        x_label = "Episode" if self.x_axis_mode == "episodes" else "Step"

        # Plot 1: Average Reward
        ax = axes[0][0]
        ax.plot(step, rew, alpha=0.3, label="Avg Reward")
        ax.plot(step, rew_ma, label=f"MA (w={self.smooth_window})", color="tab:green")
        ax.set_title("Average Reward")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 2: QoS Satisfaction
        ax = axes[0][1]
        ax.plot(step, qos, alpha=0.3, color="tab:purple", label="QoS Rate")
        ax.plot(step, qos_ma, color="tab:purple", label=f"MA (w={self.smooth_window})")
        ax.set_title("QoS Satisfaction (Mean)")
        ax.set_xlabel(x_label)
        ax.set_ylabel("QoS Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 3: Return
        ax = axes[0][2]
        ax.plot(step, ret, alpha=0.3, label="Return")
        ax.plot(step, ret_ma, label=f"MA (w={self.smooth_window})", color="tab:red")
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
        ax.plot(step, cap, alpha=0.3, label="Capacity (Mbps)", color="tab:cyan")
        ax.plot(step, cap_ma, label=f"MA (w={self.smooth_window})", color="tab:blue")
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
        self._safe_save(fig)


class NetworkMetricsPlotter(BasePlotter):
    """Plots neural network training metrics (loss, Q-values, gradients, etc.)."""

    def __init__(
        self,
        enabled: bool = True,
        plot_interval: int = 100,
        out_path: str = "Result/network_metrics.png",
        smooth_window: int = 50,
        x_axis_mode: str = "episodes",
    ):
        super().__init__(enabled, plot_interval, out_path, smooth_window)
        self.x_axis_mode = x_axis_mode if x_axis_mode in ("steps", "episodes") else "episodes"

        self._fig = None
        self._axes = None
        self._episode_count = 0 

        self.history: Dict[str, List[float]] = {
            "step": [],
            "loss": [],
            "mean_q": [],
            "max_q": [],
            "min_q": [],
            "q_std": [],
            "grad_norm": [],
            "grad_clipped": [],
            "target_q_mean": [],
            "td_error": [],
        }

    def update(
        self,
        step: int,
        loss: Optional[float] = None,
        mean_q: Optional[float] = None,
        max_q: Optional[float] = None,
        min_q: Optional[float] = None,
        q_std: Optional[float] = None,
        grad_norm: Optional[float] = None,
        grad_clipped: Optional[float] = None,
        target_q_mean: Optional[float] = None,
        td_error: Optional[float] = None,
    ):
        """Record metrics (rendering controlled by x_axis_mode)."""
        if not self.enabled or loss is None:
            return

        self._update_count += 1
        self.history["step"].append(step)
        self.history["loss"].append(loss)
        self.history["mean_q"].append(mean_q if mean_q is not None else 0.0)
        self.history["max_q"].append(max_q if max_q is not None else 0.0)
        self.history["min_q"].append(min_q if min_q is not None else 0.0)
        self.history["q_std"].append(q_std if q_std is not None else 0.0)
        self.history["grad_norm"].append(grad_norm if grad_norm is not None else 0.0)
        self.history["grad_clipped"].append(grad_clipped if grad_clipped is not None else 0.0)
        self.history["target_q_mean"].append(target_q_mean if target_q_mean is not None else 0.0)
        self.history["td_error"].append(td_error if td_error is not None else 0.0)

        # In steps mode, render based on update count
        if self.x_axis_mode == "steps" and self._should_render():
            self._render()

    def on_episode_end(self, episode: int):
        """Call at end of each episode to trigger rendering in episode mode."""
        if not self.enabled or self.x_axis_mode != "episodes":
            return
        
        self._episode_count = episode + 1
        if self._episode_count % self.plot_interval == 0:
            self._render()

    def _render(self):
        if plt is None or not self.history["step"]:
            return

        steps = self.history["step"]

        if self._fig is None:
            self._fig, self._axes = plt.subplots(3, 3, figsize=(15, 12))
        else:
            for row in self._axes:
                for ax in row:
                    ax.clear()

        fig, axes = self._fig, self._axes
        fig.suptitle("Neural Network Training Metrics", fontsize=14, fontweight="bold")

        # Row 1: Loss metrics
        # 1. TD Loss
        ax = axes[0, 0]
        loss_data = self.history["loss"]
        loss_ma = self._moving_average(loss_data)
        ax.plot(steps, loss_data, alpha=0.3, color="blue", label="Raw")
        ax.plot(steps, loss_ma, color="blue", linewidth=2, label="Smoothed")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("TD Loss (Huber)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. TD Error
        ax = axes[0, 1]
        td_data = self.history["td_error"]
        td_ma = self._moving_average(td_data)
        ax.plot(steps, td_data, alpha=0.3, color="orange")
        ax.plot(steps, td_ma, color="orange", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("TD Error")
        ax.set_title("Mean Absolute TD Error")
        ax.grid(True, alpha=0.3)

        # 3. Gradient Norm
        ax = axes[0, 2]
        grad_data = self.history["grad_norm"]
        grad_ma = self._moving_average(grad_data)
        ax.plot(steps, grad_data, alpha=0.3, color="red")
        ax.plot(steps, grad_ma, color="red", linewidth=2)
        ax.axhline(y=10.0, color="black", linestyle="--", alpha=0.5, label="Clip threshold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gradient Norm")
        ax.set_title("Gradient Norm (before clipping)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Row 2: Q-Value statistics
        # 4. Mean Q-Value
        ax = axes[1, 0]
        mean_q_data = self.history["mean_q"]
        mean_q_ma = self._moving_average(mean_q_data)
        ax.plot(steps, mean_q_data, alpha=0.3, color="green")
        ax.plot(steps, mean_q_ma, color="green", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Q")
        ax.set_title("Mean Q-Value")
        ax.grid(True, alpha=0.3)

        # 5. Q-Value Range
        ax = axes[1, 1]
        max_q = self.history["max_q"]
        min_q = self.history["min_q"]
        ax.fill_between(steps, min_q, max_q, alpha=0.3, color="purple", label="Q Range")
        ax.plot(steps, mean_q_ma, color="purple", linewidth=2, label="Mean Q")
        ax.set_xlabel("Step")
        ax.set_ylabel("Q-Value")
        ax.set_title("Q-Value Range (Min/Max)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 6. Q-Value Std
        ax = axes[1, 2]
        q_std_data = self.history["q_std"]
        q_std_ma = self._moving_average(q_std_data)
        ax.plot(steps, q_std_data, alpha=0.3, color="teal")
        ax.plot(steps, q_std_ma, color="teal", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Q Std")
        ax.set_title("Q-Value Standard Deviation")
        ax.grid(True, alpha=0.3)

        # Row 3: Target network and clipping
        # 7. Target Q Mean
        ax = axes[2, 0]
        target_q_data = self.history["target_q_mean"]
        target_q_ma = self._moving_average(target_q_data)
        ax.plot(steps, target_q_data, alpha=0.3, color="brown")
        ax.plot(steps, target_q_ma, color="brown", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Target Q Mean")
        ax.set_title("Target Network Mean Q")
        ax.grid(True, alpha=0.3)

        # 8. Policy vs Target
        ax = axes[2, 1]
        ax.plot(steps, mean_q_ma, label="Policy Q", linewidth=2)
        ax.plot(steps, target_q_ma, label="Target Q", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Q-Value")
        ax.set_title("Policy vs Target Q-Values")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # 9. Gradient Clipping Rate
        ax = axes[2, 2]
        clip_data = self.history["grad_clipped"]
        clip_ma = self._moving_average(clip_data)
        clip_rate = [v * 100 for v in clip_ma]
        ax.plot(steps, clip_rate, color="darkred", linewidth=2)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Step")
        ax.set_ylabel("Clipping Rate (%)")
        ax.set_title(f"Gradient Clipping Rate (window={self.smooth_window})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._safe_save(fig)

    def close(self):
        self.finalize()
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None


__all__ = ["BasePlotter", "RealTimeStepPlotter", "NetworkMetricsPlotter"]