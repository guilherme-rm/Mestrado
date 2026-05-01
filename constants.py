"""Global constants and default configuration values.

This module contains constants that don't need to be in the JSON config files.
Import these values where needed instead of hardcoding them.
"""

# ============== PERFORMANCE SETTINGS ==============
# Set to True for faster training (disables expensive features)
FAST_MODE = False

# Step logging throttle - higher = fewer CSV writes, better performance
# For large configs, use 50-100
DEFAULT_STEP_LOG_THROTTLE = 10 if not FAST_MODE else 100

# Shared networks mode: all agents share one policy/target network
# - True: Single network, batched updates (faster but less specialization)
# - False: Independent networks per agent (original UARA-DRL behavior)
# WARNING: Enabling this changes learning dynamics - agents won't specialize
SHARED_AGENT_NETWORKS = False

# Use a tensor-backed replay buffer to keep samples on-device and reduce
# per-step host->device transfer overhead.
USE_TENSOR_REPLAY_BUFFER = True

# ============== PLOT SETTINGS ==============
NETWORK_PLOT_ENABLED = True and not FAST_MODE
NETWORK_PLOT_INTERVAL = 50
NETWORK_PLOT_SMOOTH_WINDOW = 100
NETWORK_PLOT_FILENAME = "network_metrics.png"
NETWORK_PLOT_FIGSIZE = (14, 10)
NETWORK_PLOT_DPI = 120

DEFAULT_PLOT_SMOOTH_WINDOW = 100
DEFAULT_PLOT_INTERVAL = 50 if not FAST_MODE else 100
GRAD_CLIP_VALUE = 1.0

DEFAULT_MIN_MEMORY_FOR_LEARNING = 1000

EPS_MAX_DECAY = 0.995
EPS_MAX_MIN = 0.1

# Telecom network topology plot - expensive due to interference graph computation
TELECOM_PLOT_ENABLED = True and not FAST_MODE
TELECOM_PLOT_INTERVAL = 10 if not FAST_MODE else 50  # Plot every N episodes
TELECOM_PLOT_FILENAME = "network_topology.png"
TELECOM_PLOT_FIGSIZE = (12, 10)
TELECOM_PLOT_DPI = 100

# Resource (CPU/GPU) metrics plotting configuration
RESOURCE_PLOT_ENABLED = True and not FAST_MODE
RESOURCE_PLOT_INTERVAL = 50  # Sample every N steps
RESOURCE_PLOT_SMOOTH_WINDOW = 100
RESOURCE_PLOT_FILENAME = "resource_metrics.png"
RESOURCE_PLOT_FIGSIZE = (12, 8)
RESOURCE_PLOT_DPI = 120

# Deferred plotting mode: if True, plots are only rendered at the end of training
# instead of being updated during training. This reduces I/O overhead.
DEFERRED_PLOTTING = True


# Set to True to use GNN-based observation encoding, False for original flat obs
GNN_ENABLED = False

# Use TransformerConv-backed graph encoder when GNN is enabled.
# When False, uses the existing GCN/GAT/SAGE selection.
GNN_TRANSFORMER_ENABLED = False

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


# ============== ENVIRONMENT SELECTION ==============
# Select telecom model used by the training pipeline.
# Supported values: "hetnet" (existing behavior), "cell_free" (new model)
ENVIRONMENT_TYPE = "hetnet"
