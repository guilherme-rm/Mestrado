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
    return bool(val)


def _get_str(cfg: Any, key: str, default: str) -> str:
    val = getattr(cfg, key, None)
    if val is None:
        return str(default)
    return str(val)


def apply_runtime_settings(opt: Any, sce: Any) -> Dict[str, Any]:
    """Apply runtime behavior settings from opt/sce config objects.

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

    environment_type = _get_str(opt, "environment_type", constants.ENVIRONMENT_TYPE).strip().lower()

    mobility_enabled = _get_bool(opt, "mobility_enabled", mobility.MOBILITY_ENABLED)

    # Apply to shared runtime constants for backward compatibility across modules.
    constants.FAST_MODE = fast_mode
    constants.NETWORK_PLOT_ENABLED = network_plot_enabled
    constants.TELECOM_PLOT_ENABLED = telecom_plot_enabled
    constants.RESOURCE_PLOT_ENABLED = resource_plot_enabled
    constants.DEFERRED_PLOTTING = deferred_plotting
    constants.GNN_ENABLED = gnn_enabled
    constants.GNN_TRANSFORMER_ENABLED = gnn_transformer_enabled
    constants.GNN_OBSERVATION_MODE = gnn_observation_mode
    constants.ENVIRONMENT_TYPE = environment_type

    # Keep legacy derived default coherent.
    constants.DEFAULT_STEP_LOG_THROTTLE = int(getattr(opt, "step_log_throttle", constants.DEFAULT_STEP_LOG_THROTTLE))

    mobility.MOBILITY_ENABLED = mobility_enabled

    return {
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
