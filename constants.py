"""Global constants and default configuration values.

This module contains constants that don't need to be in the JSON config files.
Import these values where needed instead of hardcoding them.
"""

NETWORK_PLOT_ENABLED = True
NETWORK_PLOT_INTERVAL = 50
NETWORK_PLOT_SMOOTH_WINDOW = 100
NETWORK_PLOT_FILENAME = "network_metrics.png"
NETWORK_PLOT_FIGSIZE = (14, 10)
NETWORK_PLOT_DPI = 120

DEFAULT_PLOT_SMOOTH_WINDOW = 100
DEFAULT_PLOT_INTERVAL = 50
GRAD_CLIP_VALUE = 1.0

DEFAULT_MIN_MEMORY_FOR_LEARNING = 1000
