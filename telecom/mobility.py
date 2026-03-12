"""Mobility model with Random Waypoint and time-varying hotspots.

This module implements UE mobility using a Random Waypoint model with
optional attraction to dynamically generated hotspots. Hotspots themselves
move and have limited lifetimes, creating a realistic time-varying
environment.

Design:
  - All randomness uses numpy.random.Generator for reproducibility
  - Constants defined here (no config files needed)
  - Hotspots die when they leave coverage and respawn elsewhere
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from telecom.scenario import Scenario

MOBILITY_ENABLED = False

UE_SPEED = 15.0  # meters per episode (increased for visible movement)
UE_PAUSE_EPISODES = 0  # Episodes to pause at destination
CONSTRAIN_TO_COVERAGE = True  # Keep UEs within BS coverage

HOTSPOT_ENABLED = True
HOTSPOT_ATTRACTION_PROBABILITY = 0.6  # Probability UE picks hotspot destination
HOTSPOT_COUNT_RANGE = (3, 8)  # Random number of hotspots (min, max)
HOTSPOT_RADIUS_RANGE = (20.0, 80.0)  # Hotspot radius range in meters
HOTSPOT_WEIGHT_RANGE = (0.5, 2.0)  # Weight range for selection probability
HOTSPOT_SPEED = 0.3  # Hotspot movement speed (slower than UEs)
HOTSPOT_LIFETIME_RANGE = (50, 200)  # Episodes before hotspot dies and respawns

MOBILITY_SEED: Optional[int] = None

@dataclass
class Hotspot:
    """A dynamic hotspot that attracts UE movement."""
    position: np.ndarray  # (x, y)
    radius: float
    weight: float
    lifetime: int  # Episodes remaining
    velocity: np.ndarray  # Movement direction
    
    @property
    def x(self) -> float:
        return float(self.position[0])
    
    @property
    def y(self) -> float:
        return float(self.position[1])


@dataclass
class UEMobilityState:
    """Tracks mobility state for a single UE."""
    position: np.ndarray  # Current (x, y)
    destination: Optional[np.ndarray] = None  # Target (x, y)
    pause_remaining: int = 0  # Episodes to wait at destination
    
    def has_destination(self) -> bool:
        return self.destination is not None
    
    def at_destination(self, threshold: float = 1.0) -> bool:
        if self.destination is None:
            return True
        dist = np.linalg.norm(self.destination - self.position)
        return dist < threshold


class MobilityManager:
    """Manages UE mobility and time-varying hotspots.
    
    Usage:
        manager = MobilityManager(scenario, num_ues=30, seed=42)
        
        # At episode start:
        manager.update_episode()
        
        # Get UE positions:
        for i in range(num_ues):
            pos = manager.get_ue_position(i)
    """
    
    def __init__(
        self,
        scenario: Scenario,
        num_ues: int,
        seed: Optional[int] = None,
    ):
        """Initialize mobility manager.
        
        Args:
            scenario: Telecom scenario with base stations
            num_ues: Number of UEs to manage
            seed: Random seed for reproducibility (None = random)
        """
        self.scenario = scenario
        self.num_ues = num_ues
        self.seed = seed if seed is not None else MOBILITY_SEED
        
        # Initialize RNG
        self.rng = np.random.default_rng(self.seed)
        
        # Cache BS info
        self._bs_list = scenario.Get_BaseStations()
        self._coverage_bounds = self._compute_coverage_bounds()
        
        # Initialize hotspots
        self.hotspots: List[Hotspot] = []
        if HOTSPOT_ENABLED:
            self._initialize_hotspots()
        
        # Initialize UE states
        self.ue_states: List[UEMobilityState] = []
        self._initialize_ue_states()
        
        self.current_episode = 0
    
    def _compute_coverage_bounds(self) -> Tuple[float, float, float, float]:
        """Compute bounding box of all BS coverage areas.
        
        Returns:
            (x_min, x_max, y_min, y_max)
        """
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')
        
        for bs in self._bs_list:
            loc = bs.get_location()
            r = bs.radius
            x_min = min(x_min, loc[0] - r)
            x_max = max(x_max, loc[0] + r)
            y_min = min(y_min, loc[1] - r)
            y_max = max(y_max, loc[1] + r)
        
        return (x_min, x_max, y_min, y_max)
    
    def _is_within_coverage(self, position: np.ndarray) -> bool:
        """Check if position is within any BS coverage."""
        for bs in self._bs_list:
            loc = np.array(bs.get_location())
            dist = np.linalg.norm(position - loc)
            if dist <= bs.radius:
                return True
        return False
    
    def _random_position_in_coverage(self) -> np.ndarray:
        """Generate random position within BS coverage."""
        x_min, x_max, y_min, y_max = self._coverage_bounds
        
        for _ in range(100):  # Max attempts
            pos = np.array([
                self.rng.uniform(x_min, x_max),
                self.rng.uniform(y_min, y_max)
            ])
            if self._is_within_coverage(pos):
                return pos
        
        # Fallback: place near a random BS
        bs = self.rng.choice(self._bs_list)
        loc = np.array(bs.get_location())
        r = bs.radius * self.rng.random()
        theta = self.rng.uniform(-np.pi, np.pi)
        return loc + r * np.array([np.cos(theta), np.sin(theta)])
    
    def _initialize_hotspots(self):
        """Create initial set of random hotspots."""
        n_hotspots = self.rng.integers(
            HOTSPOT_COUNT_RANGE[0], 
            HOTSPOT_COUNT_RANGE[1] + 1
        )
        
        for _ in range(n_hotspots):
            self.hotspots.append(self._create_hotspot())
    
    def _create_hotspot(self) -> Hotspot:
        """Create a new random hotspot within coverage."""
        position = self._random_position_in_coverage()
        radius = self.rng.uniform(*HOTSPOT_RADIUS_RANGE)
        weight = self.rng.uniform(*HOTSPOT_WEIGHT_RANGE)
        lifetime = self.rng.integers(*HOTSPOT_LIFETIME_RANGE)
        
        # Random velocity direction
        angle = self.rng.uniform(-np.pi, np.pi)
        velocity = HOTSPOT_SPEED * np.array([np.cos(angle), np.sin(angle)])
        
        return Hotspot(
            position=position,
            radius=radius,
            weight=weight,
            lifetime=lifetime,
            velocity=velocity,
        )
    
    def _update_hotspots(self):
        """Move hotspots and handle lifecycle (death/respawn)."""
        surviving = []
        
        for hotspot in self.hotspots:
            # Decrement lifetime
            hotspot.lifetime -= 1
            
            # Move hotspot
            new_pos = hotspot.position + hotspot.velocity
            
            # Check if still in coverage and alive
            if hotspot.lifetime > 0 and self._is_within_coverage(new_pos):
                hotspot.position = new_pos
                surviving.append(hotspot)
            else:
                # Die and respawn elsewhere
                surviving.append(self._create_hotspot())
        
        self.hotspots = surviving
    
    def _select_hotspot_weighted(self) -> Optional[Hotspot]:
        """Select a hotspot using weighted random selection."""
        if not self.hotspots:
            return None
        
        weights = np.array([h.weight for h in self.hotspots])
        probs = weights / weights.sum()
        idx = self.rng.choice(len(self.hotspots), p=probs)
        return self.hotspots[idx]
    
    def _initialize_ue_states(self):
        """Initialize mobility state for all UEs."""
        self.ue_states = []
        for _ in range(self.num_ues):
            pos = self._random_position_in_coverage()
            state = UEMobilityState(position=pos.copy())
            self.ue_states.append(state)
    
    def _pick_destination(self, ue_state: UEMobilityState) -> np.ndarray:
        """Pick next destination for a UE (hotspot-biased RWM)."""
        # Decide if attracted to hotspot
        if (HOTSPOT_ENABLED 
            and self.hotspots 
            and self.rng.random() < HOTSPOT_ATTRACTION_PROBABILITY):
            # Pick hotspot-based destination
            hotspot = self._select_hotspot_weighted()
            if hotspot:
                # Random point within hotspot
                r = hotspot.radius * np.sqrt(self.rng.random())
                theta = self.rng.uniform(-np.pi, np.pi)
                dest = hotspot.position + r * np.array([np.cos(theta), np.sin(theta)])
                
                # Verify in coverage
                if self._is_within_coverage(dest):
                    return dest
        
        # Standard RWM: random point in coverage
        return self._random_position_in_coverage()
    
    def _move_ue(self, ue_state: UEMobilityState):
        """Move UE toward its destination."""
        if ue_state.pause_remaining > 0:
            ue_state.pause_remaining -= 1
            return
        
        if not ue_state.has_destination():
            ue_state.destination = self._pick_destination(ue_state)
        
        if ue_state.at_destination(threshold=UE_SPEED):
            # Arrived at destination
            ue_state.position = ue_state.destination.copy()
            ue_state.destination = None
            ue_state.pause_remaining = UE_PAUSE_EPISODES
            return
        
        # Move toward destination
        direction = ue_state.destination - ue_state.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            step = min(UE_SPEED, distance)
            new_pos = ue_state.position + direction * step
            
            # Verify still in coverage
            if CONSTRAIN_TO_COVERAGE and not self._is_within_coverage(new_pos):
                # Pick new destination instead
                ue_state.destination = self._pick_destination(ue_state)
            else:
                ue_state.position = new_pos

    
    def update_episode(self):
        """Update mobility state at the start of a new episode.
        
        Call this at the beginning of each episode to:
        - Move hotspots
        - Move UEs toward their destinations
        """
        if not MOBILITY_ENABLED:
            return
        
        self.current_episode += 1
        
        # Update hotspots first
        if HOTSPOT_ENABLED:
            self._update_hotspots()
        
        # Move each UE
        for ue_state in self.ue_states:
            self._move_ue(ue_state)
    
    def get_ue_position(self, ue_index: int) -> np.ndarray:
        """Get current position of a UE.
        
        Args:
            ue_index: UE index (0 to num_ues-1)
            
        Returns:
            Position as numpy array [x, y]
        """
        return self.ue_states[ue_index].position.copy()
    
    def set_ue_position(self, ue_index: int, position: np.ndarray):
        """Set UE position (used for initial placement from agent)."""
        self.ue_states[ue_index].position = position.copy()
        self.ue_states[ue_index].destination = None
    
    def get_hotspots(self) -> List[Hotspot]:
        """Get current hotspots for visualization."""
        return self.hotspots.copy()
    
    def get_active_hotspots(self) -> List[Hotspot]:
        """Get active hotspots (alias for get_hotspots, for plotter compatibility)."""
        return self.get_hotspots()
    
    def reset(self, seed: Optional[int] = None):
        """Reset mobility state (for new trial).
        
        Args:
            seed: New random seed (None keeps current)
        """
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
        self.current_episode = 0
        self.hotspots = []
        if HOTSPOT_ENABLED:
            self._initialize_hotspots()
        self._initialize_ue_states()


__all__ = [
    "MobilityManager",
    "Hotspot", 
    "UEMobilityState",
    "MOBILITY_ENABLED",
    "MOBILITY_SEED",
]
