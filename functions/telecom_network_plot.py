"""Telecom network topology real-time plotting utilities.

Provides a plotting class for visualizing the network topology including
base stations (MBS, PBS, FBS), user equipment (UE/agents) positions,
hotspots for mobility, and interference edges between UEs.

Updates a single PNG file that can be monitored during training.

Design goals:
  - Non-intrusive: if matplotlib is missing, plotting is silently disabled.
  - Throttled: only redraw every `plot_interval` steps to avoid overhead.
  - Headless-friendly: uses Agg backend.
  - Auto-scaling: visual elements scale based on number of UEs/BSs.
  - Supports UE mobility visualization with hotspots.
  - Shows interference edges between UEs on same channel.

Usage:
    from functions.telecom_network_plot import TelecomNetworkPlotter
    plotter = TelecomNetworkPlotter(
        scenario=scenario,
        agents=agents,
        enabled=True,
        plot_interval=10,
        out_path="Result/run/network_topology.png",
    )
    # Each step:
    plotter.update(step=global_step, actions=actions_list)
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from telecom.scenario import Scenario
from telecom.mobility import MobilityManager, Hotspot
from telecom.interference import InterferenceEdge
from rl.agent import Agent

# Lazy import matplotlib; handle absence gracefully
try:
    import matplotlib
    matplotlib.use("Agg")  # Force non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
except Exception:  # pragma: no cover - if matplotlib not present
    plt = None  # type: ignore
    Circle = None  # type: ignore
    Line2D = None  # type: ignore


# Default configuration
TELECOM_PLOT_ENABLED = True
TELECOM_PLOT_INTERVAL = 1  # Update every step by default
TELECOM_PLOT_FILENAME = "network_topology.png"
TELECOM_PLOT_FIGSIZE = (12, 10)
TELECOM_PLOT_DPI = 100

# Color scheme for base stations
BS_COLORS = {
    "MBS": "#E74C3C",  # Red for Macro BS
    "PBS": "#3498DB",  # Blue for Pico BS
    "FBS": "#2ECC71",  # Green for Femto BS
}

# Color for UEs
UE_COLOR = "#9B59B6"  # Purple for User Equipment
UE_CONNECTED_COLOR = "#F39C12"  # Orange when connected

# Hotspot colors
HOTSPOT_FILL_COLOR = "#F1C40F"  # Yellow
HOTSPOT_EDGE_COLOR = "#D4AC0D"  # Darker yellow
HOTSPOT_ALPHA = 0.25

# Interference edge colors (subtle, less prominent than connections)
INTERFERENCE_EDGE_COLOR = "#E74C3C"  # Red
INTERFERENCE_EDGE_ALPHA_MIN = 0.1
INTERFERENCE_EDGE_ALPHA_MAX = 0.3

# UE movement trace
UE_TRACE_ENABLED = True
UE_TRACE_LENGTH = 10  # Number of past positions to show
UE_TRACE_COLOR = "#9B59B6"  # Purple (same as UE)
UE_TRACE_ALPHA = 0.3
UE_TRACE_LINE_WIDTH = 1.0

# Thresholds for auto-simplification
CLUTTER_THRESHOLD_LABELS = 15  # Hide labels if total elements > this
CLUTTER_THRESHOLD_COVERAGE = 10  # Hide coverage circles if BSs > this
CLUTTER_THRESHOLD_CONNECTIONS = 30  # Thin connections if UEs > this


class TelecomNetworkPlotter:
    """Real-time plotter for telecom network topology."""

    def __init__(
        self,
        scenario: Scenario,
        agents: List[Agent],
        enabled: bool = TELECOM_PLOT_ENABLED,
        plot_interval: int = TELECOM_PLOT_INTERVAL,
        out_path: str = TELECOM_PLOT_FILENAME,
        show_coverage: bool = True,
        show_connections: bool = True,
        show_labels: bool = True,
        auto_simplify: bool = True,
        mobility_manager: Optional[MobilityManager] = None,
        show_hotspots: bool = True,
        show_interference: bool = True,
    ):
        """Initialize the telecom network plotter.

        Args:
            scenario: The telecom scenario containing base stations.
            agents: List of agents (UEs) to plot.
            enabled: Whether plotting is enabled.
            plot_interval: Number of steps between plot updates.
            out_path: Path to save the plot image.
            show_coverage: Whether to show BS coverage circles.
            show_connections: Whether to show UE-BS connections.
            show_labels: Whether to show element labels (M0, U1, etc.).
            auto_simplify: Automatically reduce clutter for large networks.
            mobility_manager: MobilityManager for hotspot visualization (optional).
            show_hotspots: Whether to show hotspot attraction zones.
            show_interference: Whether to show interference edges between UEs.
        """
        self.scenario = scenario
        self.agents = agents
        self.enabled = enabled and plt is not None
        self.plot_interval = max(1, int(plot_interval))
        self.out_path = out_path
        self.show_coverage = show_coverage
        self.show_connections = show_connections
        self.show_labels = show_labels
        self.auto_simplify = auto_simplify
        self.mobility_manager = mobility_manager
        self.show_hotspots = show_hotspots
        self.show_interference = show_interference
        
        # Cache for current connections (updated externally)
        self._current_actions: Optional[List[int]] = None
        self._current_step: int = 0
        self._current_episode: int = 0
        self._interference_edges: List[InterferenceEdge] = []
        
        # UE position history for trace visualization
        self._ue_traces: List[List[np.ndarray]] = [[] for _ in range(len(agents))]
        
        if self.enabled:
            os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)

    def update(
        self,
        step: int,
        episode: int = 0,
        actions: Optional[List[int]] = None,
        interference_edges: Optional[List[InterferenceEdge]] = None,
    ):
        """Update the network topology plot.

        Args:
            step: Current global step or step in episode.
            episode: Current episode number.
            actions: List of action indices for each agent (BS_idx * nChannel + Ch_idx).
            interference_edges: List of InterferenceEdge objects for visualization.
        """
        if not self.enabled:
            return
        
        self._current_step = step
        self._current_episode = episode
        self._current_actions = actions
        self._interference_edges = interference_edges or []
        
        # Record UE positions for trace
        if UE_TRACE_ENABLED:
            for i, agent in enumerate(self.agents):
                pos = agent.location.cpu().numpy().copy()
                self._ue_traces[i].append(pos)
                # Limit trace length
                if len(self._ue_traces[i]) > UE_TRACE_LENGTH:
                    self._ue_traces[i] = self._ue_traces[i][-UE_TRACE_LENGTH:]
        
        # Throttle rendering
        if step % self.plot_interval != 0 and step > 0:
            return
        
        self._render()

    def _get_bs_positions(self):
        """Extract base station positions and types."""
        bs_list = self.scenario.Get_BaseStations()
        positions = []
        types = []
        radii = []
        
        for bs in bs_list:
            loc = bs.get_location()
            positions.append(loc)
            types.append(bs.bs_type)
            radii.append(bs.radius)
        
        return positions, types, radii

    def _get_ue_positions(self):
        """Extract UE (agent) positions."""
        positions = []
        for agent in self.agents:
            loc = agent.location.cpu().numpy()
            positions.append(loc)
        return positions

    def _get_connections(self):
        """Determine UE-BS connections from actions.
        
        Returns:
            List of (ue_idx, bs_idx) tuples.
        """
        if self._current_actions is None:
            return []
        
        connections = []
        nChannel = self.scenario.sce.nChannel
        
        for ue_idx, action in enumerate(self._current_actions):
            if action is not None:
                bs_idx = int(action) // nChannel
                connections.append((ue_idx, bs_idx))
        
        return connections

    def _get_hotspots(self) -> List[Hotspot]:
        """Get active hotspots from mobility manager.
        
        Returns:
            List of active Hotspot objects.
        """
        if self.mobility_manager is None:
            return []
        return self.mobility_manager.get_active_hotspots()

    def _get_visual_params(self, n_bs: int, n_ue: int) -> dict:
        """Calculate visual parameters based on network size.
        
        Auto-scales marker sizes, line widths, and toggles features
        to reduce clutter for large networks.
        """
        total = n_bs + n_ue
        
        # Base sizes - larger UEs for visibility
        params = {
            "show_labels": False,  # Always hide labels for cleaner plot
            "show_coverage": True,  # Always show BS coverage circles
            "bs_size": 50,  # Dots for BS
            "ue_size": 80,  # Larger markers for UE (increased)
            "connection_alpha": 0.9,  # Strong connection visibility
            "connection_width": 2.5,  # Thicker connection lines
        }
        
        if not self.auto_simplify:
            return params
        
        # Scale down markers and thin connections based on total elements
        if total > 20:
            scale = max(0.5, 20 / total)
            params["bs_size"] = int(50 * scale)
            params["ue_size"] = int(80 * scale)  # Keep UEs relatively larger
            params["connection_alpha"] = max(0.6, 0.9 * scale)
            params["connection_width"] = max(1.5, 2.5 * scale)
        
        return params

    def _render(self):
        """Render the network topology plot."""
        if plt is None:
            return
        
        # Get positions
        bs_positions, bs_types, bs_radii = self._get_bs_positions()
        ue_positions = self._get_ue_positions()
        connections = self._get_connections() if self.show_connections else []
        hotspots = self._get_hotspots() if self.show_hotspots else []
        
        # Get auto-scaled visual parameters
        n_bs = len(bs_positions)
        n_ue = len(ue_positions)
        vp = self._get_visual_params(n_bs, n_ue)
        
        # Create figure
        plt.close("all")
        fig, ax = plt.subplots(1, 1, figsize=TELECOM_PLOT_FIGSIZE)
        
        # Calculate plot bounds
        all_x = [p[0] for p in bs_positions] + [p[0] for p in ue_positions]
        all_y = [p[1] for p in bs_positions] + [p[1] for p in ue_positions]
        
        if all_x and all_y:
            margin = max(bs_radii) * 0.2 if bs_radii else 100
            x_min, x_max = min(all_x) - margin, max(all_x) + margin
            y_min, y_max = min(all_y) - margin, max(all_y) + margin
            
            # Ensure aspect ratio is reasonable
            x_range = x_max - x_min
            y_range = y_max - y_min
            max_range = max(x_range, y_range)
            
            # Center and expand to square
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            ax.set_xlim(x_center - max_range / 2 - 50, x_center + max_range / 2 + 50)
            ax.set_ylim(y_center - max_range / 2 - 50, y_center + max_range / 2 + 50)
        
        # Draw hotspots (underneath everything else) - zorder=-1
        if self.show_hotspots and hotspots:
            for hotspot in hotspots:
                hotspot_circle = Circle(
                    (hotspot.position[0], hotspot.position[1]),
                    hotspot.radius,
                    fill=True,
                    facecolor=HOTSPOT_FILL_COLOR,
                    edgecolor=HOTSPOT_EDGE_COLOR,
                    alpha=HOTSPOT_ALPHA,
                    linewidth=2,
                    linestyle="-",
                    zorder=-1,
                )
                ax.add_patch(hotspot_circle)
        
        # Draw coverage circles (if enabled and not auto-hidden)
        if vp["show_coverage"]:
            for pos, bs_type, radius in zip(bs_positions, bs_types, bs_radii):
                color = BS_COLORS.get(bs_type, "gray")
                # Filled circle with light shade
                fill_circle = Circle(
                    (pos[0], pos[1]),
                    radius,
                    fill=True,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.15,
                    zorder=0,
                )
                ax.add_patch(fill_circle)
                # Dashed outline circle (stronger)
                outline_circle = Circle(
                    (pos[0], pos[1]),
                    radius,
                    fill=False,
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    alpha=0.7,
                    zorder=1,
                )
                ax.add_patch(outline_circle)
        
        # Draw interference edges (between UEs on same channel)
        if self.show_interference and self._interference_edges and len(ue_positions) > 0:
            # Calculate alpha scaling based on interference strength
            if self._interference_edges:
                max_strength = max(e.strength_db for e in self._interference_edges)
                min_strength = min(e.strength_db for e in self._interference_edges)
                strength_range = max_strength - min_strength if max_strength > min_strength else 1.0
                
                for edge in self._interference_edges:
                    if edge.ue_i < len(ue_positions) and edge.ue_j < len(ue_positions):
                        ue_i_pos = ue_positions[edge.ue_i]
                        ue_j_pos = ue_positions[edge.ue_j]
                        
                        # Scale alpha based on interference strength
                        normalized_strength = (edge.strength_db - min_strength) / strength_range
                        alpha = INTERFERENCE_EDGE_ALPHA_MIN + \
                                normalized_strength * (INTERFERENCE_EDGE_ALPHA_MAX - INTERFERENCE_EDGE_ALPHA_MIN)
                        
                        ax.plot(
                            [ue_i_pos[0], ue_j_pos[0]],
                            [ue_i_pos[1], ue_j_pos[1]],
                            color=INTERFERENCE_EDGE_COLOR,
                            linewidth=1.5,
                            alpha=alpha,
                            linestyle="-",
                            zorder=1.5,
                        )
        
        # Draw connections (if enabled and actions provided)
        connected_ues = set()
        if self.show_connections and connections:
            for ue_idx, bs_idx in connections:
                if ue_idx < len(ue_positions) and bs_idx < len(bs_positions):
                    ue_pos = ue_positions[ue_idx]
                    bs_pos = bs_positions[bs_idx]
                    ax.plot(
                        [ue_pos[0], bs_pos[0]],
                        [ue_pos[1], bs_pos[1]],
                        color="#D35400",  # Darker orange for better visibility
                        linewidth=vp["connection_width"],
                        alpha=vp["connection_alpha"],
                        zorder=2,
                    )
                    connected_ues.add(ue_idx)
        
        # Draw base stations as simple colored dots
        for i, (pos, bs_type) in enumerate(zip(bs_positions, bs_types)):
            color = BS_COLORS.get(bs_type, "gray")
            
            ax.scatter(
                pos[0], pos[1],
                c=color,
                marker="o",
                s=vp["bs_size"],
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )
        
        # Draw UE movement traces (before UEs so they appear underneath)
        if UE_TRACE_ENABLED:
            for i, trace in enumerate(self._ue_traces):
                if len(trace) >= 2:
                    # Draw line connecting past positions
                    xs = [p[0] for p in trace]
                    ys = [p[1] for p in trace]
                    ax.plot(
                        xs, ys,
                        color=UE_TRACE_COLOR,
                        linewidth=UE_TRACE_LINE_WIDTH,
                        alpha=UE_TRACE_ALPHA,
                        linestyle="-",
                        zorder=3.5,
                    )
                    # Draw small dots at past positions (fading)
                    for j, pos in enumerate(trace[:-1]):  # Exclude current position
                        fade_alpha = UE_TRACE_ALPHA * (j + 1) / len(trace)
                        ax.scatter(
                            pos[0], pos[1],
                            c=UE_TRACE_COLOR,
                            marker="o",
                            s=vp["ue_size"] * 0.3,
                            alpha=fade_alpha,
                            zorder=3.5,
                        )
        
        # Draw UEs (agents) as markers
        for i, pos in enumerate(ue_positions):
            color = UE_CONNECTED_COLOR if i in connected_ues else UE_COLOR
            ax.scatter(
                pos[0], pos[1],
                c=color,
                marker="o",
                s=vp["ue_size"],
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )
        
        # Create legend
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=BS_COLORS["MBS"],
                   markersize=6, markeredgecolor="black", label=f"MBS ({self.scenario.sce.nMBS})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=BS_COLORS["PBS"],
                   markersize=6, markeredgecolor="black", label=f"PBS ({self.scenario.sce.nPBS})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=BS_COLORS["FBS"],
                   markersize=6, markeredgecolor="black", label=f"FBS ({self.scenario.sce.nFBS})"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=UE_CONNECTED_COLOR,
                   markersize=6, markeredgecolor="black", label=f"UE ({len(self.agents)})"),
        ]
        
        if self.show_connections and connections:
            legend_elements.append(
                Line2D([0], [0], color="#D35400", linewidth=2,
                       label="Connection")
            )
        
        # Add hotspot legend entry
        if self.show_hotspots and hotspots:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=HOTSPOT_FILL_COLOR,
                       markersize=10, markeredgecolor=HOTSPOT_EDGE_COLOR, alpha=HOTSPOT_ALPHA,
                       label=f"Hotspot ({len(hotspots)})")
            )
        
        # Add interference edge legend entry
        if self.show_interference and self._interference_edges:
            legend_elements.append(
                Line2D([0], [0], color=INTERFERENCE_EDGE_COLOR, linewidth=1.5,
                       alpha=0.3, label=f"Interference ({len(self._interference_edges)})")
            )
        
        # Add UE trace legend entry
        if UE_TRACE_ENABLED and any(len(t) >= 2 for t in self._ue_traces):
            legend_elements.append(
                Line2D([0], [0], color=UE_TRACE_COLOR, linewidth=UE_TRACE_LINE_WIDTH,
                       alpha=UE_TRACE_ALPHA, label="UE Trace")
            )
        
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
        )
        
        # Labels and title
        ax.set_xlabel("X Position (m)", fontsize=11)
        ax.set_ylabel("Y Position (m)", fontsize=11)
        ax.set_title(
            f"Network Topology | Episode {self._current_episode}",
            fontsize=13,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        
        # Save figure
        fig.tight_layout()
        try:
            fig.savefig(self.out_path, dpi=TELECOM_PLOT_DPI)
        except Exception as e:
            print(f"Warning: Failed to save network topology plot: {e}")
        finally:
            plt.close(fig)

    def close(self):
        """Clean up resources (placeholder for future use)."""
        pass


__all__ = ["TelecomNetworkPlotter"]
