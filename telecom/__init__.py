"""Telecom system modeling package.

Contains entities related to the wireless network scenario such as
base stations, propagation models, geometry utilities, and mobility.

Public API:
    from telecom.scenario import Scenario
    from telecom.base_station import BS
    from telecom.mobility import MobilityManager
    from telecom.interference import compute_interference_graph
"""

from .base_station import BS  # noqa: F401
from .scenario import Scenario  # noqa: F401
from .mobility import MobilityManager, Hotspot  # noqa: F401
from .interference import compute_interference_graph, InterferenceEdge  # noqa: F401

__all__ = [
    "BS", 
    "Scenario", 
    "MobilityManager", 
    "Hotspot",
    "compute_interference_graph",
    "InterferenceEdge",
]
