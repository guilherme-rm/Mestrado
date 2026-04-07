"""Resource utilization (CPU/GPU) real-time plotting utilities.

Provides a plotting class for visualizing system resource usage during training,
including CPU utilization, memory usage, and GPU metrics (if available).

Design goals:
  - Non-intrusive: if dependencies are missing, plotting is silently disabled.
  - Throttled: only redraw every `plot_interval` steps to avoid overhead.
  - Headless-friendly: uses Agg backend.
  - GPU-aware: automatically detects and monitors CUDA devices.

Usage:
    from functions.resource_metrics_plot import ResourceMetricsPlotter
    plotter = ResourceMetricsPlotter(enabled=True, plot_interval=50)
    plotter.update(step=global_step)  # Automatically samples current usage
"""

from __future__ import annotations

import os
from typing import List, Dict, Optional

from constants import (
    RESOURCE_PLOT_ENABLED,
    RESOURCE_PLOT_INTERVAL,
    RESOURCE_PLOT_SMOOTH_WINDOW,
    RESOURCE_PLOT_FILENAME,
    RESOURCE_PLOT_FIGSIZE,
    RESOURCE_PLOT_DPI,
)

# Lazy import matplotlib; handle absence gracefully
try:
    import matplotlib
    matplotlib.use("Agg")  # Force non-interactive backend
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# Optional GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional CPU/memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def get_cpu_percent() -> float:
    """Get current CPU utilization percentage."""
    if PSUTIL_AVAILABLE:
        return psutil.cpu_percent(interval=None)
    return 0.0


def get_memory_percent() -> float:
    """Get current memory utilization percentage."""
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().percent
    return 0.0


def get_gpu_memory_used_mb() -> Optional[float]:
    """Get GPU memory used in MB (CUDA only).
    
    Uses memory_reserved() to get total memory held by PyTorch's caching allocator,
    which is more representative of actual GPU memory usage than memory_allocated().
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        # memory_reserved shows all memory held by the caching allocator
        # memory_allocated only shows memory for active tensors (often 0 between operations)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        if reserved > 0:
            return reserved
        # Fallback to allocated if reserved is 0 (shouldn't happen normally)
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return None


def get_gpu_memory_total_mb() -> Optional[float]:
    """Get total GPU memory in MB (CUDA only)."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    return None


def get_gpu_utilization() -> Optional[float]:
    """Get GPU utilization percentage using nvidia-smi (if available)."""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split('\n')[0])
    except Exception:
        pass
    return None


class ResourceMetricsPlotter:
    """Real-time plotter for system resource utilization metrics."""

    def __init__(
        self,
        enabled: bool = RESOURCE_PLOT_ENABLED,
        plot_interval: int = RESOURCE_PLOT_INTERVAL,
        out_path: str = RESOURCE_PLOT_FILENAME,
        smooth_window: int = RESOURCE_PLOT_SMOOTH_WINDOW,
        x_axis_mode: str = "steps",
        deferred: bool = False,
    ):
        """Initialize the resource metrics plotter.

        Args:
            enabled: Whether plotting is enabled.
            plot_interval: Number of steps between plot updates.
            out_path: Path to save the plot image.
            smooth_window: Window size for moving average smoothing.
            x_axis_mode: 'steps' or 'episodes' for x-axis labeling.
            deferred: If True, only render at the end (call render_final()).
        """
        self.enabled = enabled and plt is not None
        self.plot_interval = max(1, int(plot_interval))
        self.out_path = out_path
        self.smooth_window = max(1, int(smooth_window))
        self.x_axis_mode = (
            x_axis_mode if x_axis_mode in ("steps", "episodes") else "steps"
        )
        self.deferred = deferred

        self.history: Dict[str, List[float]] = {
            "step": [],
            "cpu_percent": [],
            "memory_percent": [],
            "gpu_memory_mb": [],
            "gpu_memory_percent": [],
            "gpu_utilization": [],
        }

        self._has_data = False
        self._gpu_total_mb = get_gpu_memory_total_mb()
        self._gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()

        if self.enabled:
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)

    def sample(self) -> Dict[str, Optional[float]]:
        """Sample current resource utilization.
        
        Returns:
            Dictionary with current resource metrics.
        """
        cpu = get_cpu_percent()
        mem = get_memory_percent()
        gpu_mem = get_gpu_memory_used_mb()
        gpu_util = get_gpu_utilization()
        
        gpu_mem_pct = None
        if gpu_mem is not None and self._gpu_total_mb:
            gpu_mem_pct = (gpu_mem / self._gpu_total_mb) * 100
        
        return {
            "cpu_percent": cpu,
            "memory_percent": mem,
            "gpu_memory_mb": gpu_mem,
            "gpu_memory_percent": gpu_mem_pct,
            "gpu_utilization": gpu_util,
        }

    def update(
        self,
        step: int,
        cpu_percent: Optional[float] = None,
        memory_percent: Optional[float] = None,
        gpu_memory_mb: Optional[float] = None,
        gpu_utilization: Optional[float] = None,
    ):
        """Update metrics history and optionally render plot.
        
        If metrics are not provided, they will be sampled automatically.

        Args:
            step: Current global step.
            cpu_percent: CPU utilization percentage (sampled if None).
            memory_percent: Memory utilization percentage (sampled if None).
            gpu_memory_mb: GPU memory used in MB (sampled if None).
            gpu_utilization: GPU utilization percentage (sampled if None).
        """
        if not self.enabled:
            return

        # Auto-sample if not provided
        if cpu_percent is None or memory_percent is None:
            sampled = self.sample()
            cpu_percent = cpu_percent if cpu_percent is not None else sampled["cpu_percent"]
            memory_percent = memory_percent if memory_percent is not None else sampled["memory_percent"]
            gpu_memory_mb = gpu_memory_mb if gpu_memory_mb is not None else sampled["gpu_memory_mb"]
            gpu_utilization = gpu_utilization if gpu_utilization is not None else sampled["gpu_utilization"]

        self._has_data = True
        self.history["step"].append(step)
        self.history["cpu_percent"].append(cpu_percent or 0.0)
        self.history["memory_percent"].append(memory_percent or 0.0)
        
        # GPU metrics (may be None if no GPU)
        self.history["gpu_memory_mb"].append(gpu_memory_mb if gpu_memory_mb is not None else 0.0)
        
        gpu_mem_pct = None
        if gpu_memory_mb is not None and self._gpu_total_mb:
            gpu_mem_pct = (gpu_memory_mb / self._gpu_total_mb) * 100
        self.history["gpu_memory_percent"].append(gpu_mem_pct if gpu_mem_pct is not None else 0.0)
        self.history["gpu_utilization"].append(gpu_utilization if gpu_utilization is not None else 0.0)

        # Skip rendering if deferred mode
        if self.deferred:
            return

        # Throttle rendering
        if step % self.plot_interval != 0 and step > 0:
            return
        self._render()

    def _moving_average(self, data: List[float]) -> List[float]:
        """Compute moving average with window size."""
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

    def _render(self):
        """Render the resource metrics plot to file."""
        if plt is None or not self._has_data:
            return

        step = self.history["step"]
        if not step:
            return

        cpu = self.history["cpu_percent"]
        mem = self.history["memory_percent"]
        gpu_mem = self.history["gpu_memory_mb"]
        gpu_mem_pct = self.history["gpu_memory_percent"]
        gpu_util = self.history["gpu_utilization"]

        cpu_ma = self._moving_average(cpu)
        mem_ma = self._moving_average(mem)
        gpu_util_ma = self._moving_average(gpu_util)
        gpu_mem_ma = self._moving_average(gpu_mem)

        plt.close("all")
        
        # Determine plot layout based on GPU availability
        if self._gpu_available:
            fig, axes = plt.subplots(2, 2, figsize=RESOURCE_PLOT_FIGSIZE)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes = list(axes)
        
        x_label = "Episode" if self.x_axis_mode == "episodes" else "Step"

        # Plot 1: CPU Utilization
        ax = axes[0]
        ax.plot(step, cpu, alpha=0.3, label="CPU %", color="tab:blue")
        ax.plot(step, cpu_ma, label=f"CPU MA (w={self.smooth_window})", color="darkblue")
        ax.set_title("CPU Utilization")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        # Plot 2: System Memory
        ax = axes[1]
        ax.plot(step, mem, alpha=0.3, label="RAM %", color="tab:green")
        ax.plot(step, mem_ma, label=f"RAM MA (w={self.smooth_window})", color="darkgreen")
        ax.set_title("System Memory Utilization")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        if self._gpu_available:
            # Plot 3: GPU Memory
            ax = axes[2]
            ax.plot(step, gpu_mem, alpha=0.3, label="GPU Memory (MB)", color="tab:red")
            ax.plot(step, gpu_mem_ma, label=f"GPU Mem MA (w={self.smooth_window})", color="darkred")
            if self._gpu_total_mb:
                ax.axhline(y=self._gpu_total_mb, color='gray', linestyle='--', alpha=0.5, 
                          label=f"Total: {self._gpu_total_mb:.0f} MB")
            ax.set_title("GPU Memory Usage")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Memory (MB)")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)

            # Plot 4: GPU Utilization
            ax = axes[3]
            ax.plot(step, gpu_util, alpha=0.3, label="GPU Util %", color="tab:orange")
            ax.plot(step, gpu_util_ma, label=f"GPU Util MA (w={self.smooth_window})", color="darkorange")
            ax.set_title("GPU Utilization")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Percentage (%)")
            ax.set_ylim(0, 105)
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)

        fig.suptitle("System Resource Utilization", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(self.out_path, dpi=RESOURCE_PLOT_DPI)

    def render_final(self):
        """Force a final render (for deferred mode)."""
        if self.enabled:
            self._render()

    def close(self):
        """Clean up resources."""
        # Force final render if we have data
        if self.enabled and self._has_data:
            self._render()

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of resource usage.
        
        Returns:
            Dictionary with mean/max values for each metric.
        """
        if not self._has_data:
            return {}
        
        def _mean(data):
            return sum(data) / len(data) if data else 0.0
        
        def _max(data):
            return max(data) if data else 0.0
        
        return {
            "cpu_mean": _mean(self.history["cpu_percent"]),
            "cpu_max": _max(self.history["cpu_percent"]),
            "memory_mean": _mean(self.history["memory_percent"]),
            "memory_max": _max(self.history["memory_percent"]),
            "gpu_memory_mean_mb": _mean(self.history["gpu_memory_mb"]) if self._gpu_available else None,
            "gpu_memory_max_mb": _max(self.history["gpu_memory_mb"]) if self._gpu_available else None,
            "gpu_utilization_mean": _mean(self.history["gpu_utilization"]) if self._gpu_available else None,
            "gpu_utilization_max": _max(self.history["gpu_utilization"]) if self._gpu_available else None,
        }


__all__ = ["ResourceMetricsPlotter", "get_cpu_percent", "get_memory_percent", 
           "get_gpu_memory_used_mb", "get_gpu_utilization"]
