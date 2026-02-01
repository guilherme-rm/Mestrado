"""Training utilities for the RL agents."""

from __future__ import annotations

# Import math for exponential decay
import math
from dataclasses import dataclass
from typing import Sequence, List, Optional, Dict

import torch

from .agent import Agent, TrainMetrics


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
    step_idx: int,
    opt,
) -> Optional[Dict[str, float]]:
    """Store transitions and perform learning step.
    
    Returns:
        Aggregated training metrics across all agents that learned,
        or None if no agent learned this step.
    """
    # Get nupdate for hard target update period
    nupdate = getattr(opt, "nupdate", 50)
    
    all_metrics: List[TrainMetrics] = []

    for i, ag in enumerate(agents):
        # 1. Store transition
        ag.Save_Transition(state, actions[i], next_state, rewards[i], scenario)

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


def create_agents(opt, sce, scenario, device) -> List[Agent]:
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
