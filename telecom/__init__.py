"""Telecom system modeling package.

Contains entities related to the wireless network scenario such as
base stations, propagation models, and geometry utilities.

Public API:
    from telecom.scenario import Scenario
    from telecom.base_station import BS
"""

from .base_station import BS  # noqa: F401
from .scenario import Scenario  # noqa: F401

__all__ = ["BS", "Scenario"]
