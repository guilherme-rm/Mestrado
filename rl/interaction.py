"""Shared observation, movement, and joint-allocation operations.

Movement changes geometry only. QoS and rewards are evaluated only after a
new joint allocation; historical feedback remains part of the observation.
"""
from typing import Sequence
import torch
from .observations import enriched_observations


class EnvironmentInteraction:
    """Shared policy/environment boundary used by training and evaluation."""

    def __init__(self, agents, scenario, manager=None):
        self.agents = agents
        self.scenario = scenario
        self.manager = manager

    def observe(self, state, actions=None, capacity=None):
        if self.manager is not None and self.manager.enabled:
            return self.manager.encode(self.agents, actions)
        return enriched_observations(self.agents, self.scenario, state, actions, capacity)

    @torch.inference_mode()
    def greedy_actions(self, state, actions=None, capacity=None):
        obs = self.observe(state, actions, capacity)
        return torch.stack([agent.model_policy(obs[i] if obs is not None else state).argmax().reshape(())
                            for i, agent in enumerate(self.agents)])

    def move_users(self, mover):
        mover.update_episode()
        for i, agent in enumerate(self.agents):
            agent.set_location(mover.get_ue_position(i))

    def apply_joint_action(self, actions, state):
        return compute_rewards_and_next_state(self.agents, actions, state, self.scenario)


def movement_due(completed_decisions, cadence, horizon):
    """Whether to move after a decision while another decision remains."""
    return cadence > 0 and completed_decisions > 0 and completed_decisions % cadence == 0 and completed_decisions < horizon


def observations(agents, scenario, state, manager=None, actions=None, capacity=None):
    return EnvironmentInteraction(agents, scenario, manager).observe(state, actions, capacity)


@torch.inference_mode()
def greedy_actions(agents, scenario, state, manager=None, actions=None, capacity=None):
    return EnvironmentInteraction(agents, scenario, manager).greedy_actions(state, actions, capacity)


def move_users(agents, mover):
    """Move once without recalculating allocations or rewards."""
    EnvironmentInteraction(agents, None).move_users(mover)


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
