"""Agent definition (UE) tying together network, memory, and learning logic."""
from __future__ import annotations

from random import random, uniform, choice, randrange
from typing import Sequence

import numpy as np
from numpy import pi
import torch
# Import Adam optimizer
from torch import optim
import torch.nn as nn
from torch.nn import functional as F

from .memory import ReplayMemory, Transition
from .networks import DNN


class Agent:
    def __init__(self, opt, sce, scenario, index: int, device: torch.device):
        self.opt = opt
        self.sce = sce
        self.id = index
        self.device = device
        self.location = self._set_location(scenario)
        self.memory = ReplayMemory(opt.capacity)
        
        self.model_policy = DNN(opt, sce, scenario).to(self.device)
        self.model_target = DNN(opt, sce, scenario).to(self.device)
        self.model_target.load_state_dict(self.model_policy.state_dict())
        self.model_target.eval()
        
        # Initialize Adam optimizer
        self.optimizer = optim.Adam(
            params=self.model_policy.parameters(),
            lr=opt.learning_rate,
        )

    # --- geometry / initialization ---
    def _set_location(self, scenario) -> torch.Tensor:
        # (Implementation remains the same)
        Loc_MBS, _, _ = scenario.BS_Location()
        Loc_agent = np.zeros(2)
        LocM = choice(Loc_MBS)
        r = self.sce.rMBS * random()
        theta = uniform(-pi, pi)
        Loc_agent[0] = LocM[0] + r * np.cos(theta)
        Loc_agent[1] = LocM[1] + r * np.sin(theta)
        return torch.tensor(Loc_agent, device=self.device, dtype=torch.float32)

    def Get_Location(self) -> torch.Tensor:
        return self.location

    # --- action selection ---
    def Select_Action(self, state: torch.Tensor, scenario, eps: float) -> torch.Tensor:
        L = scenario.BS_Number()
        K = self.sce.nChannel
        if random() < eps:  # explore
            return torch.tensor([[randrange(L * K)]], dtype=torch.long, device=self.device)
        with torch.no_grad():  # exploit
            q = self.model_policy(state.to(self.device))
            return q.argmax(dim=0, keepdim=True).view(1, 1)

    # --- reward computation ---
    def Get_Reward(self, action_all: torch.Tensor, action_i: torch.Tensor, state: torch.Tensor, scenario):
        # (Environment simulation logic)
        BS_list = scenario.Get_BaseStations()
        K = self.sce.nChannel

        BS_selected = int(action_i // K)
        Ch_selected = int(action_i % K)
        
        # Basic validation for action index
        if BS_selected >= len(BS_list):
             return 0, torch.tensor(self.sce.negative_cost, device=self.device, dtype=torch.float32), torch.tensor(0.0, device=self.device)

        BS_location = torch.tensor(BS_list[BS_selected].Get_Location(), device=self.device, dtype=torch.float32)
        Loc_diff = BS_location - self.location
        distance = torch.sqrt(Loc_diff[0] ** 2 + Loc_diff[1] ** 2)
        d_val = max(float(distance.item()), 1.0)
        Rx_power = BS_list[BS_selected].Receive_Power(d_val)

        if Rx_power <= 1e-20: # Small threshold
            return 0, torch.tensor(self.sce.negative_cost, device=self.device, dtype=torch.float32), torch.tensor(0.0, device=self.device)

        Interference = 0.0
        for i in range(self.opt.nagents):
            if int(action_all[i] % K) == Ch_selected:
                sel_bs_i = int(action_all[i] // K)
                if sel_bs_i >= len(BS_list): continue

                BS_loc_i = torch.tensor(BS_list[sel_bs_i].Get_Location(), device=self.device, dtype=torch.float32)
                diff_i = BS_loc_i - self.location
                d_i = torch.sqrt(diff_i[0] ** 2 + diff_i[1] ** 2)
                d_i_val = max(float(d_i.item()), 1.0)
                Interference += BS_list[sel_bs_i].Receive_Power(d_i_val)
        
        # Ensure non-negative interference
        Interference = max(0.0, Interference - Rx_power)

        Noise = 10 ** (self.sce.N0 / 10) * self.sce.BW
        SINR = Rx_power / (Interference + Noise)
        spec_eff = np.log2(1.0 + SINR) if SINR > 0 else 0.0
        capacity_bps = self.sce.BW * spec_eff
        capacity_mbps = capacity_bps / 1e6
        cap_tensor = torch.tensor(float(capacity_mbps), device=self.device)
        
        if SINR >= 10 ** (self.sce.QoS_thr / 10):
            # Positive reward based on profit config (defaults to 1.0)
            reward_val = float(getattr(self.sce, 'profit', 1.0))
            return 1, torch.tensor(reward_val, device=self.device, dtype=torch.float32), cap_tensor
        
        # Negative reward if QoS not met
        return 0, torch.tensor(self.sce.negative_cost, device=self.device, dtype=torch.float32), cap_tensor

    # --- replay / learning ---
    def Save_Transition(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor, scenario):
        # Store transition with batch dimension added
        self.memory.Push(state.unsqueeze(0), action.view(1, 1), next_state.unsqueeze(0), reward.view(1))

    def Soft_Target_Update(self, tau: float = 0.005):
        """Performs soft update of the target network parameters (Polyak averaging).
        θ_target = τ*θ_policy + (1 - τ)*θ_target
        """
        target_state_dict = self.model_target.state_dict()
        policy_state_dict = self.model_policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = policy_state_dict[key]*tau + target_state_dict[key]*(1.0-tau)
        self.model_target.load_state_dict(target_state_dict)

    def Optimize_Model(self):
        """Performs a single optimization step using Double DQN."""
        if len(self.memory) < self.opt.batch_size:
            return

        # Check if minimum memory threshold for learning is reached
        if hasattr(self.opt, 'min_memory_for_learning') and len(self.memory) < self.opt.min_memory_for_learning:
            return

        transitions = self.memory.Sample(self.opt.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        # Compute Q(s,a)
        q_pred = self.model_policy(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) using Double DQN
        with torch.no_grad():
            # 1. Select best action a' using policy network
            next_actions = self.model_policy(next_state_batch).argmax(1, keepdim=True)
            # 2. Evaluate that action using target network
            q_next = self.model_target(next_state_batch).gather(1, next_actions)
            # Compute the target Q values
            target_q = reward_batch.unsqueeze(1) + self.opt.gamma * q_next

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_pred, target_q)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (using clip_grad_norm_ for stability)
        torch.nn.utils.clip_grad_norm_(self.model_policy.parameters(), max_norm=10.0)
        self.optimizer.step()


__all__ = ["Agent"]
