"""Runtime settings resolved from configuration files.

This module makes config JSON files the primary source of runtime behavior,
minimizing hardcoded flags and manual constant edits.
"""

from __future__ import annotations

from typing import Any, Dict

import constants
from telecom import mobility


def _get_bool(cfg: Any, key: str, default: bool) -> bool:
    val = getattr(cfg, key, None)
    if val is None:
        return bool(default)
    if not isinstance(val, bool):
        raise ValueError(f"{key} must be a JSON boolean")
    return val


def _get_str(cfg: Any, key: str, default: str) -> str:
    val = getattr(cfg, key, None)
    if val is None:
        return str(default)
    return str(val)


def apply_runtime_settings(opt: Any, sce: Any) -> Dict[str, Any]:
    """Resolve runtime behavior settings from opt/sce config objects.

    Precedence:
    1) explicit values in opt/sce
    2) existing constants defaults
    """
    fast_mode = _get_bool(opt, "fast_mode", constants.FAST_MODE)

    enable_plot = _get_bool(opt, "enable_plot", True)
    network_plot_enabled = _get_bool(opt, "network_plot_enabled", enable_plot and not fast_mode)
    telecom_plot_enabled = _get_bool(opt, "telecom_plot_enabled", enable_plot and not fast_mode)
    resource_plot_enabled = _get_bool(opt, "resource_plot_enabled", enable_plot and not fast_mode)

    deferred_plotting = _get_bool(opt, "deferred_plotting", constants.DEFERRED_PLOTTING)

    gnn_enabled = _get_bool(opt, "gnn_enabled", constants.GNN_ENABLED)
    gnn_transformer_enabled = _get_bool(opt, "gnn_transformer_enabled", constants.GNN_TRANSFORMER_ENABLED)
    if gnn_transformer_enabled:
        gnn_enabled = True

    gnn_observation_mode = _get_str(opt, "gnn_observation_mode", constants.GNN_OBSERVATION_MODE)

    # Resolve environment type: opt > sce._env > constant default
    _env_type_raw = getattr(opt, "environment_type", None)
    if _env_type_raw is None:
        _env_type_raw = getattr(sce, "_env", None)
    if _env_type_raw is None:
        _env_type_raw = constants.ENVIRONMENT_TYPE
    environment_type = str(_env_type_raw).strip().lower()
    # Write back to opt so that all downstream consumers see the resolved value
    opt.environment_type = environment_type

    mobility_enabled = _get_bool(opt, "mobility_enabled", mobility.MOBILITY_ENABLED)

    resolved = {
        "enable_plot": enable_plot,
        "fast_mode": fast_mode,
        "network_plot_enabled": network_plot_enabled,
        "telecom_plot_enabled": telecom_plot_enabled,
        "resource_plot_enabled": resource_plot_enabled,
        "deferred_plotting": deferred_plotting,
        "gnn_enabled": gnn_enabled,
        "gnn_transformer_enabled": gnn_transformer_enabled,
        "gnn_observation_mode": gnn_observation_mode,
        "environment_type": environment_type,
        "mobility_enabled": mobility_enabled,
    }

    for key, value in resolved.items():
        setattr(opt, key, value)
    return resolved
