"""Training utilities for the RL agents."""

from __future__ import annotations

# Import math for exponential decay
import math
from dataclasses import dataclass
from typing import Sequence, List, Optional, Dict

import numpy as np
import torch

from constants import (
    GNN_ENABLED,
    GNN_HIDDEN_DIM,
    GNN_OUTPUT_DIM,
    GNN_NUM_LAYERS,
    GNN_USE_ATTENTION,
    GNN_OBSERVATION_MODE,
    GNN_INCLUDE_INTERFERENCE_EDGES,
    GNN_HETEROGENEOUS,
)
from .agent import Agent, TrainMetrics


@dataclass
class TrainingContext:
    state: torch.Tensor
    next_state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    qos: torch.Tensor

def select_actions(
    agents: Sequence[Agent], state: torch.Tensor, scenario, eps: float,
    gnn_observations: Dict[int, torch.Tensor] = None,
) -> torch.Tensor:
    """Select actions for all agents.
    
    Args:
        agents: List of agents
        state: Flat state tensor (used if gnn_observations is None)
        scenario: Telecom scenario
        eps: Exploration epsilon
        gnn_observations: Optional dict of GNN observations per agent
        
    Returns:
        Tensor of actions for each agent
    """
    raw = []
    for i, ag in enumerate(agents):
        # Use GNN observation if available, otherwise flat state
        obs = gnn_observations[i] if gnn_observations and i in gnn_observations else state
        raw.append(ag.Select_Action(obs, scenario, eps))
    stacked = torch.stack(raw)
    return stacked.view(len(agents))


def compute_rewards_and_next_state(
    agents: Sequence[Agent], actions: torch.Tensor, state: torch.Tensor, scenario
):
    nagents = len(agents)
    rewards = torch.zeros(nagents, device=state.device)
    qos = torch.zeros(nagents, device=state.device)
    next_state = torch.zeros(nagents, device=state.device)
    capacity = torch.zeros(nagents, device=state.device)
    for i, ag in enumerate(agents):
        qos_i, reward_i, cap_i = ag.Get_Reward(actions, actions[i], state, scenario)
        qos[i] = qos_i
        rewards[i] = reward_i
        capacity[i] = cap_i
        # State representation is the QoS satisfaction of the previous step
        next_state[i] = qos_i
    return rewards, qos, next_state, capacity


def store_and_learn(
    agents: Sequence[Agent],
    state: torch.Tensor,
    actions: torch.Tensor,
    next_state: torch.Tensor,
    rewards: torch.Tensor,
    scenario,
    step_idx: int,
    opt,
    gnn_obs: Optional[Dict[int, torch.Tensor]] = None,
    gnn_obs_next: Optional[Dict[int, torch.Tensor]] = None,
) -> Optional[Dict[str, float]]:
    """Store transitions and perform learning step.
    
    Args:
        agents: List of agents
        state: Flat state tensor (used when GNN disabled)
        actions: Actions taken by each agent
        next_state: Next flat state tensor (used when GNN disabled)
        rewards: Rewards for each agent
        scenario: Network scenario
        step_idx: Current step index
        opt: Training options
        gnn_obs: Optional dict mapping agent_id to GNN observation for current state
        gnn_obs_next: Optional dict mapping agent_id to GNN observation for next state
    
    Returns:
        Aggregated training metrics across all agents that learned,
        or None if no agent learned this step.
    """
    # Get nupdate for hard target update period (handle DotDic returning None)
    nupdate = getattr(opt, "nupdate", None)
    if nupdate is None:
        nupdate = 50
    
    all_metrics: List[TrainMetrics] = []

    for i, ag in enumerate(agents):
        # 1. Store transition - use GNN observations if available
        if gnn_obs is not None and gnn_obs_next is not None:
            # GNN mode: use per-agent embeddings
            # IMPORTANT: detach() to prevent keeping computation graph in replay buffer
            agent_state = gnn_obs.get(i).detach()
            agent_next_state = gnn_obs_next.get(i).detach()
        else:
            # Flat mode: use shared state
            agent_state = state
            agent_next_state = next_state
        
        ag.Save_Transition(agent_state, actions[i], agent_next_state, rewards[i], scenario)

        # 2. Optimize policy network
        metrics = ag.Optimize_Model()
        if metrics.did_learn:
            all_metrics.append(metrics)

        # 3. Hard target update every nupdate steps (like original UARA-DRL)
        if step_idx % nupdate == 0:
            ag.Target_Update()
    
    if not all_metrics:
        return None
    
    n = len(all_metrics)
    return {
        "loss": sum(m.loss for m in all_metrics) / n,
        "mean_q": sum(m.mean_q for m in all_metrics) / n,
        "max_q": max(m.max_q for m in all_metrics),
        "min_q": min(m.min_q for m in all_metrics),
        "q_std": sum(m.q_std for m in all_metrics) / n,
        "grad_norm": sum(m.grad_norm for m in all_metrics) / n,
        "target_q_mean": sum(m.target_q_mean for m in all_metrics) / n,
        "td_error": sum(m.td_error for m in all_metrics) / n,
    }


def initialize_episode(nagents: int, device: torch.device) -> TrainingContext:
    return TrainingContext(
        state=torch.zeros(nagents, device=device),
        next_state=torch.zeros(nagents, device=device),
        actions=torch.zeros(nagents, dtype=torch.long, device=device),
        rewards=torch.zeros(nagents, device=device),
        qos=torch.zeros(nagents, device=device),
    )


def should_terminate(state: torch.Tensor, target_state: torch.Tensor) -> bool:
    return bool(torch.all(state.eq(target_state)))


def create_agents(
    opt,
    sce,
    scenario,
    device,
    mobility_manager=None,
    gnn_input_dim: int = None,
) -> List[Agent]:
    """Create agents (UEs) for the simulation.
    
    Args:
        opt: Training configuration.
        sce: Scenario configuration.
        scenario: Scenario instance.
        device: PyTorch device.
        mobility_manager: Optional MobilityManager for UE mobility support.
        gnn_input_dim: If provided, agents use this input dim (for GNN mode).
        
    Returns:
        List of Agent instances.
    """
    return [
        Agent(
            opt,
            sce,
            scenario,
            index=i,
            device=device,
            mobility_manager=mobility_manager,
            input_dim=gnn_input_dim,
        )
        for i in range(opt.nagents)
    ]


class GNNObservationManager:
    """Manages GNN-based observation encoding for MARL agents.
    
    This class wraps the GNN encoder and provides methods to:
    1. Build graph from current network state
    2. Encode observations using GNN
    3. Gracefully fall back to flat observations when GNN is disabled
    
    Usage:
        # Create once at training start
        gnn_manager = GNNObservationManager(scenario, device)
        
        # Each step, get observations
        if gnn_manager.enabled:
            gnn_obs = gnn_manager.encode(agents, actions)
            # Use gnn_obs for action selection
    """
    
    def __init__(
        self,
        scenario,
        device: torch.device,
        enabled: bool = GNN_ENABLED,
    ):
        """Initialize the GNN observation manager.
        
        Args:
            scenario: Telecom scenario with BS/channel configuration
            device: PyTorch device for computation
            enabled: Whether to use GNN encoding (from constants.GNN_ENABLED)
        """
        self.scenario = scenario
        self.device = device
        self._enabled = enabled
        self._encoder = None
        
        if enabled:
            self._init_encoder()
    
    def _init_encoder(self):
        """Lazily initialize the GNN encoder."""
        from rl.gnn import GNNObservationEncoder
        
        self._encoder = GNNObservationEncoder(
            scenario=self.scenario,
            device=self.device,
            mode=GNN_OBSERVATION_MODE,
            gnn_output_dim=GNN_OUTPUT_DIM,
            gnn_hidden_dim=GNN_HIDDEN_DIM,
            gnn_num_layers=GNN_NUM_LAYERS,
            use_attention=GNN_USE_ATTENTION,
            include_interference_edges=GNN_INCLUDE_INTERFERENCE_EDGES,
            heterogeneous=GNN_HETEROGENEOUS,
        )
    
    @property
    def enabled(self) -> bool:
        """Whether GNN observation encoding is enabled."""
        return self._enabled and self._encoder is not None
    
    @property
    def output_dim(self) -> int:
        """Dimension of GNN output observations."""
        if self._encoder is not None:
            return self._encoder.output_dim
        return 0
    
    def encode(
        self,
        agents: Sequence[Agent],
        actions: Optional[torch.Tensor] = None,
        sinrs: Optional[List[float]] = None,
    ) -> Dict[int, torch.Tensor]:
        """Encode current network state into GNN observations.
        
        Args:
            agents: List of Agent objects (for positions)
            actions: Current action indices per agent
            sinrs: SINR values per agent (optional)
            
        Returns:
            Dictionary mapping agent_id -> observation tensor
        """
        if not self.enabled:
            return {}
        
        # Extract UE positions from agents
        ue_positions = [
            agent.location.cpu().numpy() if isinstance(agent.location, torch.Tensor)
            else np.array(agent.location)
            for agent in agents
        ]
        
        # Convert actions tensor to list
        action_list = None
        if actions is not None:
            action_list = actions.tolist() if isinstance(actions, torch.Tensor) else list(actions)
        
        return self._encoder(
            ue_positions=ue_positions,
            actions=action_list,
            ue_sinrs=sinrs,
        )
    
    def get_state_for_agent(
        self,
        agent_id: int,
        gnn_observations: Dict[int, torch.Tensor],
        flat_state: torch.Tensor,
    ) -> torch.Tensor:
        """Get the appropriate state representation for an agent.
        
        If GNN is enabled, returns GNN observation. Otherwise returns flat state.
        
        Args:
            agent_id: Agent index
            gnn_observations: Dict from encode() or empty if GNN disabled
            flat_state: Original flat state tensor
            
        Returns:
            State tensor for the agent
        """
        if self.enabled and agent_id in gnn_observations:
            return gnn_observations[agent_id]
        return flat_state


def select_actions_with_gnn(
    agents: Sequence[Agent],
    gnn_manager: GNNObservationManager,
    flat_state: torch.Tensor,
    scenario,
    eps: float,
    prev_actions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select actions using GNN observations if enabled, else flat state.
    
    Args:
        agents: List of agents
        gnn_manager: GNN observation manager (may be disabled)
        flat_state: Original flat state representation
        scenario: Telecom scenario
        eps: Exploration epsilon
        prev_actions: Previous step actions (for graph construction)
        
    Returns:
        Tensor of selected actions for each agent
    """
    if gnn_manager.enabled:
        # Get GNN observations
        gnn_obs = gnn_manager.encode(agents, prev_actions)
        
        # Select actions using GNN observations
        raw_actions = []
        for i, agent in enumerate(agents):
            state = gnn_manager.get_state_for_agent(i, gnn_obs, flat_state)
            action = agent.Select_Action(state, scenario, eps)
            raw_actions.append(action)
        
        stacked = torch.stack(raw_actions)
        return stacked.view(len(agents))
    else:
        # Fall back to original flat state
        raw = [ag.Select_Action(flat_state, scenario, eps) for ag in agents]
        stacked = torch.stack(raw)
        return stacked.view(len(agents))


__all__ = [
    "TrainingContext",
    "select_actions",
    "compute_rewards_and_next_state",
    "store_and_learn",
    "initialize_episode",
    "should_terminate",
    "create_agents",
    "GNNObservationManager",
    "select_actions_with_gnn",
]
