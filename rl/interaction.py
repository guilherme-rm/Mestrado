"""Shared observation, movement, and joint-allocation operations.

Movement changes geometry only. QoS and rewards are evaluated only after a
new joint allocation; historical feedback remains part of the observation.
"""
from typing import Sequence
import torch
from .observations import enriched_observations


def observations(agents, scenario, state, manager=None, actions=None, capacity=None):
    if manager is not None and manager.enabled:
        return manager.encode(agents, actions)
    return enriched_observations(agents, scenario, state, actions, capacity)


@torch.inference_mode()
def greedy_actions(agents, scenario, state, manager=None, actions=None, capacity=None):
    obs = observations(agents, scenario, state, manager, actions, capacity)
    return torch.stack([agent.model_policy(obs[i] if obs is not None else state).argmax().reshape(())
                        for i, agent in enumerate(agents)])


def move_users(agents, mover):
    """Move once without recalculating allocations or rewards."""
    mover.update_episode()
    for i, agent in enumerate(agents):
        agent.set_location(mover.get_ue_position(i))


def compute_rewards_and_next_state(
    agents: Sequence, actions: torch.Tensor, state: torch.Tensor, scenario
):
    from rl.training import build_next_state
    nagents = len(agents)
    if hasattr(scenario, "set_current_ue_positions"):
        scenario.set_current_ue_positions([
            ag.location.detach().cpu().numpy() if isinstance(ag.location, torch.Tensor) else ag.location
            for ag in agents
        ])
    if hasattr(scenario, "begin_step"):
        scenario.begin_step()
    rewards = torch.zeros(nagents, device=state.device)
    qos = torch.zeros(nagents, device=state.device)
    capacity = torch.zeros(nagents, device=state.device)
    for i, ag in enumerate(agents):
        qos_i, reward_i, cap_i = ag.Get_Reward(actions, actions[i], state, scenario)
        qos[i] = qos_i
        rewards[i] = reward_i
        capacity[i] = cap_i
    next_state = build_next_state(qos, capacity, actions, scenario)
    cellfree_step_metrics = {}
    if hasattr(scenario, "get_visual_diagnostics"):
        cellfree_step_metrics = scenario.get_visual_diagnostics().get("step_metrics", {}) or {}
    return rewards, qos, next_state, capacity, cellfree_step_metrics
