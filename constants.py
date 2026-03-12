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

EPS_MAX_DECAY = 0.995
EPS_MAX_MIN = 0.1

TELECOM_PLOT_ENABLED = True
TELECOM_PLOT_INTERVAL = 1  
TELECOM_PLOT_FILENAME = "network_topology.png"
TELECOM_PLOT_FIGSIZE = (12, 10)
TELECOM_PLOT_DPI = 100


# Set to True to use GNN-based observation encoding, False for original flat obs
GNN_ENABLED = True

# GNN architecture parameters
GNN_HIDDEN_DIM = 64
GNN_OUTPUT_DIM = 64
GNN_NUM_LAYERS = 2
GNN_USE_ATTENTION = False
GNN_DROPOUT = 0.1

# Observation mode: "replace" (GNN only) or "augment" (GNN + flat obs)
GNN_OBSERVATION_MODE = "replace"

# Graph construction options
GNN_INCLUDE_INTERFERENCE_EDGES = True
GNN_INCLUDE_POTENTIAL_LINKS = True
GNN_HETEROGENEOUS = False  # Use separate params for UE/BS nodes
