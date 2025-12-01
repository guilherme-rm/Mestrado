"""Agent definition (UE) tying together network, memory, and learning logic."""

from __future__ import annotations

from random import random, uniform, choice, randrange
from typing import Sequence, Tuple

import numpy as np
from numpy import pi
import torch

# Import Adam optimizer
from torch import optim
import torch.nn as nn
from torch.nn import functional as F

from .memory import ReplayMemory, Transition
from .networks import DNN

class LocationManager:
    """Handles agent location initialization and access."""
    
    def __init__(self, sce, scenario, device: torch.device):
        self.sce = sce
        self.device = device
        self._location = self._initialize_location(scenario)
    
    def _initialize_location(self, scenario) -> torch.Tensor:
        Loc_MBS, _, _ = scenario.BS_Location()
        Loc_agent = np.zeros(2)
        LocM = choice(Loc_MBS)
        r = self.sce.rMBS * random()
        theta = uniform(-pi, pi)
        Loc_agent[0] = LocM[0] + r * np.cos(theta)
        Loc_agent[1] = LocM[1] + r * np.sin(theta)
        return torch.tensor(Loc_agent, device=self.device, dtype=torch.float32)
    
    @property
    def location(self) -> torch.Tensor:
        return self._location
    
class RewardCalculator:
    """Encapsulates reward computation logic."""
    
    def __init__(self, sce, opt, device: torch.device):
        self.sce = sce
        self.opt = opt
        self.device = device
    
    def compute(
        self,
        action_all: torch.Tensor,
        action_i: torch.Tensor,
        agent_location: torch.Tensor,
        scenario,
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Compute QoS satisfaction, reward, and capacity."""
        BS_list = scenario.Get_BaseStations()
        K = self.sce.nChannel

        BS_selected = int(action_i // K)
        Ch_selected = int(action_i % K)

        # Basic validation for action index
        if BS_selected >= len(BS_list):
            return self._invalid_action_result()

        rx_power = self._compute_rx_power(BS_list[BS_selected], agent_location)
        if rx_power <= 1e-20:
            return self._invalid_action_result()

        interference = self._compute_interference(
            action_all, Ch_selected, BS_list, agent_location, rx_power
        )
        
        sinr, capacity_mbps = self._compute_sinr_and_capacity(rx_power, interference)
        cap_tensor = torch.tensor(float(capacity_mbps), device=self.device)

        if sinr >= 10 ** (self.sce.QoS_thr / 10):
            reward_val = float(getattr(self.sce, "profit", 1.0))
            return (1, torch.tensor(reward_val, device=self.device, dtype=torch.float32), cap_tensor)

        return (0, torch.tensor(self.sce.negative_cost, device=self.device, dtype=torch.float32), cap_tensor)
    
    def _invalid_action_result(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        return (
            0,
            torch.tensor(self.sce.negative_cost, device=self.device, dtype=torch.float32),
            torch.tensor(0.0, device=self.device),
        )
    
    def _compute_rx_power(self, bs, agent_location: torch.Tensor) -> float:
        BS_location = torch.tensor(bs.Get_Location(), device=self.device, dtype=torch.float32)
        Loc_diff = BS_location - agent_location
        distance = torch.sqrt(Loc_diff[0] ** 2 + Loc_diff[1] ** 2)
        d_val = max(float(distance.item()), 1.0)
        return bs.Receive_Power(d_val)
    
    def _compute_interference(
        self, action_all, channel, BS_list, agent_location, rx_power
    ) -> float:
        K = self.sce.nChannel
        interference = 0.0
        
        for i in range(self.opt.nagents):
            if int(action_all[i] % K) == channel:
                sel_bs_i = int(action_all[i] // K)
                if sel_bs_i >= len(BS_list):
                    continue
                interference += self._compute_rx_power(BS_list[sel_bs_i], agent_location)
        
        return max(0.0, interference - rx_power)
    
    def _compute_sinr_and_capacity(self, rx_power: float, interference: float) -> Tuple[float, float]:
        noise = 10 ** (self.sce.N0 / 10) * self.sce.BW
        sinr = rx_power / (interference + noise)
        spec_eff = np.log2(1.0 + sinr) if sinr > 0 else 0.0
        capacity_bps = self.sce.BW * spec_eff
        return sinr, capacity_bps / 1e6
    
class DQNOptimizer:
    """Handles DQN optimization logic."""
    
    def __init__(self, model_policy, model_target, optimizer, opt, device):
        self.model_policy = model_policy
        self.model_target = model_target
        self.optimizer = optimizer
        self.opt = opt
        self.device = device
    
    def optimize(self, memory: ReplayMemory):
        """Performs a single optimization step using Double DQN."""
        if len(memory) < self.opt.batch_size:
            return

        if (hasattr(self.opt, "min_memory_for_learning") 
            and len(memory) < self.opt.min_memory_for_learning):
            return

        transitions = memory.Sample(self.opt.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        q_pred = self.model_policy(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_actions = self.model_policy(next_state_batch).argmax(1, keepdim=True)
            q_next = self.model_target(next_state_batch).gather(1, next_actions)
            target_q = reward_batch.unsqueeze(1) + self.opt.gamma * q_next

        loss = F.smooth_l1_loss(q_pred, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_policy.parameters(), max_norm=10.0)
        self.optimizer.step()
    
    def soft_update(self, tau: float = 0.005):
        """Performs soft update of the target network parameters."""
        target_state_dict = self.model_target.state_dict()
        policy_state_dict = self.model_policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = (
                policy_state_dict[key] * tau + target_state_dict[key] * (1.0 - tau)
            )
        self.model_target.load_state_dict(target_state_dict)

class ActionSelector:
    """Encapsulates action selection strategy (epsilon-greedy, Boltzmann, etc.)."""
    
    def __init__(self, strategy: str = "epsilon_greedy"):
        self.strategy = strategy
    
    def select(self, q_values: torch.Tensor, epsilon: float) -> int:
        """Select action based on Q-values and exploration parameter."""
        if self.strategy == "epsilon_greedy":
            if random() < epsilon:
                return randrange(q_values.size(-1))
            else:
                return int(q_values.argmax().item())
        elif self.strategy == "greedy":
            return int(q_values.argmax().item())
        elif self.strategy == "boltzmann":
            temperature = max(epsilon, 0.01)  # Use epsilon as temperature
            probs = torch.softmax(q_values / temperature, dim=-1)
            return int(torch.multinomial(probs, 1).item())
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
class Agent:
    
    def __init__(self, opt, sce, scenario, index: int, device: torch.device):
        self.opt = opt
        self.sce = sce
        self.id = index
        self.device = device
        
        # Composed components
        self._location_manager = LocationManager(sce, scenario, device)
        self._reward_calculator = RewardCalculator(sce, opt, device)
        self._action_selector = ActionSelector(
            strategy=getattr(opt, "action_strategy", "epsilon_greedy")
        )
        
        # Memory
        self.memory = ReplayMemory(opt.capacity)
        
        # Networks
        self.model_policy = DNN(opt, sce, scenario).to(device)
        self.model_target = DNN(opt, sce, scenario).to(device)
        self.model_target.load_state_dict(self.model_policy.state_dict())
        self.model_target.eval()
        
        # Optimizer component
        self.optimizer = optim.Adam(
            params=self.model_policy.parameters(),
            lr=opt.learning_rate,
        )
        self._dqn_optimizer = DQNOptimizer(
            self.model_policy, self.model_target, self.optimizer, opt, device
        )
        
        # Cache for action space size
        self._n_actions = scenario.BS_Number() * sce.nChannel

    @property
    def location(self) -> torch.Tensor:
        return self._location_manager.location

    def Get_Location(self) -> torch.Tensor:
        return self.location

    def Select_Action(self, state: torch.Tensor, scenario, eps: float) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.model_policy(state)
        
        action = self._action_selector.select(
            q_values, eps
        )

        return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def Get_Reward(self, action_all, action_i, state, scenario):
        return self._reward_calculator.compute(
            action_all, action_i, self.location, scenario
        )

    def Save_Transition(self, state, action, next_state, reward, scenario):
        self.memory.Push(
            state.unsqueeze(0),
            action.view(1, 1),
            next_state.unsqueeze(0),
            reward.view(1),
        )

    def Soft_Target_Update(self, tau: float = 0.005):
        self._dqn_optimizer.soft_update(tau)

    def Optimize_Model(self):
        self._dqn_optimizer.optimize(self.memory)


__all__ = ["Agent", "ActionSelector", "RewardCalculator", "DQNOptimizer", "LocationManager"]