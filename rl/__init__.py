"""Lazy public exports; importing utilities does not load training or plotting."""
from importlib import import_module

_EXPORTS = {'Transition': 'memory', 'ReplayMemory': 'memory', 'DNN': 'networks', 'Agent': 'agent', 'GNNObservationEncoder': 'gnn', 'WirelessGraphBuilder': 'gnn', 'WirelessGraph': 'gnn', 'GNNEncoder': 'gnn', 'EdgeConditionedGNN': 'gnn'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
