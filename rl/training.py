"""Training utilities for the RL agents."""

from __future__ import annotations

# Import math for exponential decay
import math
import time
from dataclasses import dataclass
from typing import Sequence, List, Optional, Dict

import torch

from torch import optim

from constants import (
    GNN_ENABLED,
    GNN_HIDDEN_DIM,
    GNN_OUTPUT_DIM,
    GNN_NUM_LAYERS,
    GNN_USE_ATTENTION,
    GNN_OBSERVATION_MODE,
    GNN_INCLUDE_INTERFERENCE_EDGES,
    GNN_HETEROGENEOUS,
    GNN_TRANSFORMER_ENABLED,
    SHARED_AGENT_NETWORKS,
    USE_TENSOR_REPLAY_BUFFER,
)
from .agent import Agent, TrainMetrics, DQNOptimizer
from .networks import DNN
from .memory import ReplayMemory, TensorReplayMemory


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
    return_timing: bool = False,
):
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
    t_total = time.perf_counter()
    forward_seconds = 0.0
    policy_seconds = 0.0

    # Check if using shared networks for batched forward pass optimization
    uses_shared = agents[0].uses_shared_networks if agents else False
    
    if uses_shared and gnn_observations is None:
        # Batched forward pass for shared networks with flat state
        # All agents see the same state, so we only need one forward pass
        # and apply action selection per agent
        t0 = time.perf_counter()
        with torch.no_grad():
            q_values = agents[0].model_policy(state)  # Single forward pass
        forward_seconds = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        if q_values.dim() == 1:
            q_batch = q_values.unsqueeze(0).expand(len(agents), -1)
        else:
            q_batch = q_values.expand(len(agents), -1)
        actions = agents[0]._action_selector.select_batch(q_batch, eps)
        policy_seconds = time.perf_counter() - t0
        if return_timing:
            return actions, {
                "select_actions_forward_seconds": forward_seconds,
                "select_actions_policy_seconds": policy_seconds,
                "select_actions_total_seconds": time.perf_counter() - t_total,
            }
        return actions
    
    elif uses_shared and gnn_observations is not None:
        # Batched forward pass for shared networks with GNN observations
        # Stack all observations and do single batched forward pass
        obs_list = [gnn_observations[i] for i in range(len(agents))]
        batched_obs = torch.stack(obs_list)  # (nagents, obs_dim)
        
        t0 = time.perf_counter()
        with torch.no_grad():
            all_q_values = agents[0].model_policy(batched_obs)  # (nagents, n_actions)
        forward_seconds = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        actions = agents[0]._action_selector.select_batch(all_q_values, eps)
        policy_seconds = time.perf_counter() - t0
        if return_timing:
            return actions, {
                "select_actions_forward_seconds": forward_seconds,
                "select_actions_policy_seconds": policy_seconds,
                "select_actions_total_seconds": time.perf_counter() - t_total,
            }
        return actions
    
    else:
        # Original per-agent forward passes for independent networks
        raw = []
        t0 = time.perf_counter()
        for i, ag in enumerate(agents):
            # Use GNN observation if available, otherwise flat state
            obs = gnn_observations[i] if gnn_observations and i in gnn_observations else state
            raw.append(ag.Select_Action(obs, scenario, eps))
        policy_seconds = time.perf_counter() - t0
        stacked = torch.stack(raw)
        actions = stacked.view(len(agents))
        if return_timing:
            return actions, {
                "select_actions_forward_seconds": 0.0,
                "select_actions_policy_seconds": policy_seconds,
                "select_actions_total_seconds": time.perf_counter() - t_total,
            }
        return actions


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
) -> tuple[Optional[Dict[str, float]], Dict[str, float]]:
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
        Tuple of:
        - Aggregated training metrics across all agents that learned,
          or None if no agent learned this step.
        - Timing breakdown dictionary for store/learn phases.
    """
    t_total = time.perf_counter()
    # Get nupdate for hard target update period (handle DotDic returning None)
    nupdate = getattr(opt, "nupdate", None)
    if nupdate is None:
        nupdate = 50
    
    # Check if using shared networks (all agents share same network)
    uses_shared = agents[0].uses_shared_networks if agents else False
    learn_every = max(1, int(getattr(opt, "learn_every", 1) or 1))
    do_optimize = (step_idx % learn_every) == 0
    parallel_opt = bool(getattr(opt, "parallel_agent_optimization", False))
    max_streams = max(1, int(getattr(opt, "parallel_opt_max_streams", 4) or 1))
    
    all_metrics: List[TrainMetrics] = []
    store_transition_seconds = 0.0
    optimize_seconds = 0.0
    target_update_seconds = 0.0
    detach_seconds = 0.0

    for i, ag in enumerate(agents):
        # 1. Store transition - use GNN observations if available
        if gnn_obs is not None and gnn_obs_next is not None:
            # GNN mode: use per-agent embeddings
            # IMPORTANT: detach() to prevent keeping computation graph in replay buffer
            t0 = time.perf_counter()
            agent_state = gnn_obs.get(i).detach()
            agent_next_state = gnn_obs_next.get(i).detach()
            detach_seconds += time.perf_counter() - t0
        else:
            # Flat mode: use shared state
            agent_state = state
            agent_next_state = next_state
        
        t0 = time.perf_counter()
        ag.Save_Transition(agent_state, actions[i], agent_next_state, rewards[i], scenario)
        store_transition_seconds += time.perf_counter() - t0

    if do_optimize:
        if uses_shared:
            # With shared networks: only optimize once (on first agent)
            t0 = time.perf_counter()
            metrics = agents[0].Optimize_Model()
            optimize_seconds += time.perf_counter() - t0
            if metrics.did_learn:
                all_metrics.append(metrics)

            # Hard target update (only once since networks are shared)
            if step_idx % nupdate == 0:
                t0 = time.perf_counter()
                agents[0].Target_Update()
                target_update_seconds += time.perf_counter() - t0
        else:
            use_parallel_streams = (
                parallel_opt
                and state.device.type == "cuda"
                and len(agents) > 1
            )

            if use_parallel_streams:
                n_streams = min(max_streams, len(agents))
                streams = [torch.cuda.Stream(device=state.device) for _ in range(n_streams)]
                metrics_buffer: List[Optional[TrainMetrics]] = [None] * len(agents)

                t0 = time.perf_counter()
                for idx, ag in enumerate(agents):
                    stream = streams[idx % n_streams]
                    with torch.cuda.stream(stream):
                        metrics_buffer[idx] = ag.Optimize_Model()
                for stream in streams:
                    stream.synchronize()
                optimize_seconds += time.perf_counter() - t0

                for metrics in metrics_buffer:
                    if metrics is not None and metrics.did_learn:
                        all_metrics.append(metrics)
            else:
                # Independent networks: optimize each agent separately
                for ag in agents:
                    t0 = time.perf_counter()
                    metrics = ag.Optimize_Model()
                    optimize_seconds += time.perf_counter() - t0
                    if metrics.did_learn:
                        all_metrics.append(metrics)

            # Hard target update every nupdate steps (like original UARA-DRL)
            if step_idx % nupdate == 0:
                t0 = time.perf_counter()
                for ag in agents:
                    ag.Target_Update()
                target_update_seconds += time.perf_counter() - t0

    timing = {
        "store_and_learn_total_seconds": time.perf_counter() - t_total,
        "store_transition_seconds": store_transition_seconds,
        "detach_seconds": detach_seconds,
        "optimize_seconds": optimize_seconds,
        "target_update_seconds": target_update_seconds,
    }

    if not all_metrics:
        return None, timing
    
    n = len(all_metrics)
    metrics = {
        "loss": sum(m.loss for m in all_metrics) / n,
        "mean_q": sum(m.mean_q for m in all_metrics) / n,
        "max_q": max(m.max_q for m in all_metrics),
        "min_q": min(m.min_q for m in all_metrics),
        "q_std": sum(m.q_std for m in all_metrics) / n,
        "grad_norm": sum(m.grad_norm for m in all_metrics) / n,
        "target_q_mean": sum(m.target_q_mean for m in all_metrics) / n,
        "td_error": sum(m.td_error for m in all_metrics) / n,
    }
    return metrics, timing


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
    shared_networks = None
    shared_memory = None
    
    if SHARED_AGENT_NETWORKS:
        # Create single shared network for all agents
        model_policy = DNN(opt, sce, scenario, input_dim=gnn_input_dim).to(device)
        model_target = DNN(opt, sce, scenario, input_dim=gnn_input_dim).to(device)
        model_target.load_state_dict(model_policy.state_dict())
        model_target.eval()
        
        momentum = opt.momentum if opt.momentum is not None else 0.0
        learning_rate = (
            getattr(opt, "learning_rate", None)
            or getattr(opt, "learningrate", None)
            or 5e-4
        )
        optimizer = optim.RMSprop(
            params=model_policy.parameters(),
            lr=learning_rate,
            momentum=momentum,
        )
        dqn_optimizer = DQNOptimizer(
            model_policy, model_target, optimizer, opt, device
        )
        shared_networks = (model_policy, model_target, optimizer, dqn_optimizer)
        
        # Create single shared replay buffer - larger capacity since all agents contribute
        # Use nagents * original_capacity to maintain similar per-agent experience storage
        shared_capacity = opt.capacity * opt.nagents
        if USE_TENSOR_REPLAY_BUFFER:
            shared_memory = TensorReplayMemory(shared_capacity)
        else:
            shared_memory = ReplayMemory(shared_capacity)
    
    return [
        Agent(
            opt,
            sce,
            scenario,
            index=i,
            device=device,
            mobility_manager=mobility_manager,
            input_dim=gnn_input_dim,
            shared_networks=shared_networks,
            shared_memory=shared_memory,
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
            conv_type="transformer" if GNN_TRANSFORMER_ENABLED else "gcn",
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
            agent.location if isinstance(agent.location, torch.Tensor)
            else torch.as_tensor(agent.location, device=self.device, dtype=torch.float32)
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
