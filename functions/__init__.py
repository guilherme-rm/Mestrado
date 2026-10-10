"""Lazy public exports; importing utilities does not load training or plotting."""
from importlib import import_module

_EXPORTS = {'RealTimeStepPlotter': 'live_plot', 'TelecomNetworkPlotter': 'telecom_network_plot', 'NetworkMetricsPlotter': 'network_metrics_plot', 'ResourceMetricsPlotter': 'resource_metrics_plot', 'RunDirectoryManager': 'logging', 'save_config': 'logging', 'save_environment_snapshot': 'logging', 'EpisodeMetricsLogger': 'logging', 'StepMetricsLogger': 'logging', 'checkpoint_agents': 'logging', 'write_summary': 'logging', 'write_experiment_summary': 'logging'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
