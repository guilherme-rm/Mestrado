"""Training utilities for the RL agents.

This consolidates helpers previously in the monolithic `functions.py` into
the RL package namespace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List

import torch

from .agent import Agent


@dataclass
class TrainingContext:
    state: torch.Tensor
    next_state: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    qos: torch.Tensor



class EpsilonScheduler:
    def __init__(self, opt):
        self.opt = opt

    def value(self, episode_idx: int, step_idx: int) -> float:
        return compute_epsilon(self.opt, episode_idx, step_idx)


def compute_epsilon(opt, episode_idx: int, step_idx: int) -> float:
    lin = opt.eps_min + opt.eps_increment * step_idx * (episode_idx + 1)
    return min(lin, opt.eps_max)


def select_actions(agents: Sequence[Agent], state: torch.Tensor, scenario, eps: float) -> torch.Tensor:
    # Each agent currently returns an action tensor shaped (1,1); we want a flat (nagents,) long tensor.
    raw = [ag.Select_Action(state, scenario, eps) for ag in agents]
    stacked = torch.stack(raw)  # shape (nagents,1,1)
    return stacked.view(len(agents))  # safe reshape without dropping nagents when =1


def compute_rewards_and_next_state(agents: Sequence[Agent], actions: torch.Tensor, state: torch.Tensor, scenario):
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
):
    for i, ag in enumerate(agents):
        ag.Save_Transition(state, actions[i], next_state, rewards[i], scenario)
        ag.Optimize_Model()
        if step_idx % opt.nupdate == 0:
            ag.Target_Update()


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
    return [Agent(opt, sce, scenario, index=i, device=device) for i in range(opt.nagents)]


__all__ = [
    "TrainingContext",
    # EpisodeLogger removed; use functions.logging.EpisodeMetricsLogger instead
    "EpsilonScheduler",
    "compute_epsilon",
    "select_actions",
    "compute_rewards_and_next_state",
    "store_and_learn",
    "initialize_episode",
    "should_terminate",
    "create_agents",
]
