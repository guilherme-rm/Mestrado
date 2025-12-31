"""Training utilities for the RL agents."""

from __future__ import annotations

# Import math for exponential decay
import math
from dataclasses import dataclass
from typing import Sequence, List, Optional
from .metrics import NetworkMetrics

import torch

from .agent import Agent


@dataclass
class TrainingContext:
    state: torch.Tensor
    next_state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    qos: torch.Tensor

def select_actions(
    agents: Sequence[Agent], state: torch.Tensor, scenario, eps: float
) -> torch.Tensor:
    raw = [ag.Select_Action(state, scenario, eps) for ag in agents]
    stacked = torch.stack(raw)  # shape (nagents,1,1)
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
    step_idx: int,  # Kept for compatibility but unused
    opt,
) -> Optional[NetworkMetrics]:
    # Get tau for soft update (defaulting to 0.005 if not set)
    tau = getattr(opt, "tau", 0.005)

    for i, ag in enumerate(agents):
        # 1. Store transition
        ag.Save_Transition(state, actions[i], next_state, rewards[i], scenario)

        # 2. Optimize policy network
        network_metrics = ag.Optimize_Model()
        

        # 3. Use Soft Target Update at every step for stable learning
        ag.Soft_Target_Update(tau=tau)

        return network_metrics


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


def create_agents(opt, sce, scenario, device) -> List[Agent]:
    
    Agent.initialize_all_locations(opt.nagents, device)
    
    return [
        Agent(opt, sce, scenario, index=i, device=device) for i in range(opt.nagents)
    ]


__all__ = [
    "TrainingContext",
    "select_actions",
    "compute_rewards_and_next_state",
    "store_and_learn",
    "initialize_episode",
    "should_terminate",
    "create_agents",
]
