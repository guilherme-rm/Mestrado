"""GPU management utilities.

Single module responsible for all GPU-related decisions:
  - Device selection (CPU / single GPU / multi-GPU)
  - CUDA fast-math and cuDNN tuning flags
  - Automatic batch-size scaling from free VRAM
  - AMP (Automatic Mixed Precision) context helpers
  - GPU memory / utilization queries

Usage
-----
Call ``GPUManager.from_opt(opt)`` once at experiment start, then pass the
resulting ``GPUManager`` instance wherever device or AMP decisions are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import pynvml


_NVML_READY = False


def _init_nvml() -> bool:
    """Initialize NVML exactly once.

    Returns False if pynvml is unavailable or NVML initialization fails.
    """
    global _NVML_READY
    if _NVML_READY:
        return True
    try:
        pynvml.nvmlInit()
        _NVML_READY = True
        return True
    except Exception:
        return False


def _device_index(device: Optional[torch.device] = None) -> int:
    """Resolve a torch device to a CUDA index (default 0)."""
    if device is not None and device.type == "cuda" and device.index is not None:
        return int(device.index)
    return 0


def _nvml_handle(device_index: int):
    if not _init_nvml():
        return None
    try:
        return pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GPU info helpers (no opt required)
# ---------------------------------------------------------------------------

def get_free_vram_gb(device: torch.device) -> float:
    """Return free VRAM in GB for a CUDA device (0.0 on CPU)."""
    if device.type != "cuda":
        return 0.0
    idx = _device_index(device)
    handle = _nvml_handle(idx)
    if handle is not None:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.free) / (1024 ** 3)
        except Exception:
            pass
    torch.cuda.synchronize(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return free_bytes / (1024 ** 3)


def get_total_vram_gb(device: torch.device) -> float:
    """Return total VRAM in GB for a CUDA device (0.0 on CPU)."""
    if device.type != "cuda":
        return 0.0
    idx = _device_index(device)
    handle = _nvml_handle(idx)
    if handle is not None:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.total) / (1024 ** 3)
        except Exception:
            pass
    _, total_bytes = torch.cuda.mem_get_info(device)
    return total_bytes / (1024 ** 3)


def get_vram_mb(device: Optional[torch.device] = None) -> float:
    """Return currently allocated GPU memory in MB (used by resource plotter)."""
    if not torch.cuda.is_available():
        return 0.0
    idx = _device_index(device)
    handle = _nvml_handle(idx)
    if handle is not None:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.used) / (1024 * 1024)
        except Exception:
            pass
    return torch.cuda.memory_allocated() / (1024 * 1024)


def get_total_vram_mb(device_index: int = 0) -> float:
    """Return total VRAM in MB for the given device index."""
    if not torch.cuda.is_available():
        return 0.0
    handle = _nvml_handle(device_index)
    if handle is not None:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.total) / (1024 * 1024)
        except Exception:
            pass
    return torch.cuda.get_device_properties(device_index).total_memory / (1024 * 1024)


def get_device_name(device_index: int = 0) -> Optional[str]:
    """Return CUDA device name, or None if CUDA is unavailable."""
    if torch.cuda.is_available() and _init_nvml():
        handle = _nvml_handle(device_index)
        if handle is not None:
            try:
                raw_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(raw_name, bytes):
                    return raw_name.decode("utf-8", errors="ignore")
                return str(raw_name)
            except Exception:
                pass
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(device_index)
    return None


def cuda_available() -> bool:
    return torch.cuda.is_available()


def nvml_available() -> bool:
    """Return whether NVML telemetry is usable."""
    return _init_nvml()


def get_gpu_utilization_percent(device_index: int = 0) -> Optional[float]:
    """Return GPU utilization percentage via NVML, if available."""
    handle = _nvml_handle(device_index)
    if handle is None:
        return None
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return None


def get_gpu_memory_used_mb(device_index: int = 0) -> Optional[float]:
    """Return GPU used memory in MB via NVML, if available."""
    handle = _nvml_handle(device_index)
    if handle is None:
        return None
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.used) / (1024 * 1024)
    except Exception:
        return None


def get_free_total_bytes(device: torch.device) -> tuple[int, int]:
    """Return (free_bytes, total_bytes) for a device, preferring NVML."""
    if device.type != "cuda":
        return 0, 0

    idx = _device_index(device)
    handle = _nvml_handle(idx)
    if handle is not None:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.free), int(info.total)
        except Exception:
            pass

    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return int(free_bytes), int(total_bytes)


# ---------------------------------------------------------------------------
# GPUManager – single source of truth for all GPU decisions
# ---------------------------------------------------------------------------

@dataclass
class DevicePlan:
    """Result of GPU device selection."""
    device: torch.device
    multi_gpu_enabled: bool = False
    multi_gpu_device_ids: List[int] = field(default_factory=list)

    def __str__(self) -> str:
        if self.multi_gpu_enabled:
            return f"multi-GPU {self.multi_gpu_device_ids}"
        return str(self.device)


class GPUManager:
    """Central manager for GPU-related configuration and runtime decisions.

    Attributes
    ----------
    device : torch.device
        Primary device for model/tensor allocation.
    multi_gpu_device_ids : list[int]
        GPU IDs selected for multi-GPU training (empty when single-GPU).
    amp_enabled : bool
        Whether automatic mixed precision is active.
    """

    def __init__(
        self,
        *,
        multi_gpu_auto: bool = True,
        multi_gpu_min_memory_gb: float = 8.0,
        use_amp: bool = True,
        batch_size_auto: bool = False,
        batch_size_vram_fraction: float = 0.6,
        batch_size_min: int = 64,
        batch_size_max: int = 4096,
        batch_size_grad_overhead: float = 5.0,
    ):
        self._multi_gpu_auto = multi_gpu_auto
        self._multi_gpu_min_memory_gb = multi_gpu_min_memory_gb
        self._use_amp_cfg = use_amp
        self._batch_size_auto = batch_size_auto
        self._batch_size_vram_fraction = batch_size_vram_fraction
        self._batch_size_min = batch_size_min
        self._batch_size_max = batch_size_max
        self._batch_size_grad_overhead = batch_size_grad_overhead

        plan = self._select_devices()
        self.device: torch.device = plan.device
        self.multi_gpu_enabled: bool = plan.multi_gpu_enabled
        self.multi_gpu_device_ids: List[int] = plan.multi_gpu_device_ids
        self.amp_enabled: bool = self._use_amp_cfg and self.device.type == "cuda"

        self._apply_fast_math()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_opt(cls, opt) -> "GPUManager":
        """Build a GPUManager from a DotDic/namespace opt object."""
        def _g(name, default):
            v = getattr(opt, name, None)
            return default if v is None else v

        return cls(
            multi_gpu_auto=bool(_g("multi_gpu_auto", True)),
            multi_gpu_min_memory_gb=float(_g("multi_gpu_min_memory_gb", 8.0)),
            use_amp=bool(_g("use_amp", True)),
            batch_size_auto=bool(_g("batch_size_auto", False)),
            batch_size_vram_fraction=float(_g("batch_size_vram_fraction", 0.6)),
            batch_size_min=int(_g("batch_size_min", 64)),
            batch_size_max=int(_g("batch_size_max", 4096)),
            batch_size_grad_overhead=float(_g("batch_size_grad_overhead", 5.0)),
        )

    @staticmethod
    def build_amp_context(opt, device: torch.device) -> tuple[bool, torch.cuda.amp.GradScaler]:
        """Build (amp_enabled, GradScaler) without creating a full GPUManager."""
        amp_cfg = getattr(opt, "use_amp", None)
        if amp_cfg is None:
            amp_cfg = getattr(opt, "amp_enabled", device.type == "cuda")
        amp_enabled = bool(amp_cfg) and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        return amp_enabled, scaler

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------

    def _select_devices(self) -> DevicePlan:
        """Determine which device(s) to use and return a DevicePlan."""
        if not torch.cuda.is_available():
            print("GPUManager: CPU (CUDA not available)")
            return DevicePlan(device=torch.device("cpu"))

        gpu_count = torch.cuda.device_count()
        if not self._multi_gpu_auto or gpu_count < 2:
            dev = torch.device("cuda:0")
            print(f"GPUManager: single GPU ({torch.cuda.get_device_name(0)})")
            return DevicePlan(device=dev)

        eligible: List[int] = []
        for gpu_id in range(gpu_count):
            handle = _nvml_handle(gpu_id)
            if handle is not None:
                try:
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_gb = float(info.total) / (1024 ** 3)
                except Exception:
                    props = torch.cuda.get_device_properties(gpu_id)
                    total_gb = props.total_memory / (1024 ** 3)
            else:
                props = torch.cuda.get_device_properties(gpu_id)
                total_gb = props.total_memory / (1024 ** 3)
            if total_gb >= self._multi_gpu_min_memory_gb:
                eligible.append(gpu_id)

        if len(eligible) >= 2:
            dev = torch.device(f"cuda:{eligible[0]}")
            print(
                f"GPUManager: multi-GPU on {eligible} "
                f"(min {self._multi_gpu_min_memory_gb:.1f} GB per GPU)"
            )
            return DevicePlan(device=dev, multi_gpu_enabled=True, multi_gpu_device_ids=eligible)

        dev = torch.device("cuda:0")
        print(
            f"GPUManager: single GPU "
            f"(only {len(eligible)}/{gpu_count} GPUs meet the "
            f"{self._multi_gpu_min_memory_gb:.1f} GB threshold)"
        )
        return DevicePlan(device=dev)

    # ------------------------------------------------------------------
    # CUDA kernel tuning
    # ------------------------------------------------------------------

    def _apply_fast_math(self) -> None:
        """Enable TF32, cuDNN benchmark, and high matmul precision on CUDA."""
        if self.device.type != "cuda":
            return
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------
    # AMP helpers
    # ------------------------------------------------------------------

    def make_grad_scaler(self) -> torch.cuda.amp.GradScaler:
        """Return a GradScaler configured for current AMP setting."""
        return torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def autocast(self):
        """Return a torch.cuda.amp.autocast context manager."""
        return torch.cuda.amp.autocast(enabled=self.amp_enabled)

    # ------------------------------------------------------------------
    # Batch-size auto-scaling
    # ------------------------------------------------------------------

    def auto_scale_batch_size(self, current_batch_size: int, state_dim: int) -> int:
        """Return a batch size fitted to the current free VRAM.

        Parameters
        ----------
        current_batch_size:
            The value from config (returned unchanged when auto-scaling is off).
        state_dim:
            Feature dimension of the state vector (used to estimate per-sample cost).

        Returns
        -------
        int
            The new batch size (may equal ``current_batch_size`` unchanged).
        """
        if not self._batch_size_auto:
            return current_batch_size
        if self.device.type != "cuda":
            print("GPUManager batch-size auto-scale: skipped (CPU device)")
            return current_batch_size

        free_bytes, total_bytes = get_free_total_bytes(self.device)
        budget = int(free_bytes * self._batch_size_vram_fraction)

        # 2 × state (float32) + 1 action (int32) + 1 reward (float32)
        bytes_per_sample = (2 * state_dim * 4 + 4 + 4) * self._batch_size_grad_overhead
        computed = int(budget // bytes_per_sample)
        new_bs = max(self._batch_size_min, min(self._batch_size_max, computed))

        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        print(
            f"GPUManager batch-size auto-scale: {current_batch_size} → {new_bs} "
            f"(free VRAM {free_gb:.2f}/{total_gb:.2f} GB, "
            f"fraction {self._batch_size_vram_fraction:.0%})"
        )
        return new_bs

    # ------------------------------------------------------------------
    # Write resolved values back to an opt object so existing logging
    # code that reads opt.multi_gpu_enabled etc. continues to work.
    # ------------------------------------------------------------------

    def apply_to_opt(self, opt) -> None:
        """Stamp resolved GPU settings back onto the opt DotDic."""
        opt["multi_gpu_enabled"] = self.multi_gpu_enabled
        opt["multi_gpu_device_ids"] = self.multi_gpu_device_ids
        opt["amp_enabled"] = self.amp_enabled

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a dict of current GPU state for logging."""
        info: dict = {
            "device": str(self.device),
            "multi_gpu_enabled": self.multi_gpu_enabled,
            "multi_gpu_device_ids": self.multi_gpu_device_ids,
            "amp_enabled": self.amp_enabled,
            "cuda_available": torch.cuda.is_available(),
            "nvml_available": nvml_available(),
        }
        if torch.cuda.is_available():
            info["cuda_device_name"] = get_device_name(_device_index(self.device))
            free_gb = get_free_vram_gb(self.device)
            total_gb = get_total_vram_gb(self.device)
            info["vram_free_gb"] = round(free_gb, 2)
            info["vram_total_gb"] = round(total_gb, 2)
            util = get_gpu_utilization_percent(_device_index(self.device))
            if util is not None:
                info["gpu_utilization_percent"] = round(util, 2)
        return info

    def __repr__(self) -> str:
        return (
            f"GPUManager(device={self.device}, "
            f"multi_gpu={self.multi_gpu_device_ids or 'off'}, "
            f"amp={self.amp_enabled})"
        )
