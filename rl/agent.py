"""Agent definition (UE) tying together network, memory, and learning logic."""

from __future__ import annotations

from random import random, uniform, choice, randrange
from typing import Tuple, Optional, Sequence

import numpy as np
from numpy import pi
import torch

# Import Adam optimizer
from torch import optim
from torch.nn import functional as F

from .memory import ReplayMemory, Transition
from .networks import DNN
from .metrics import NetworkMetrics, compute_gradient_norm
from telecom.scenario import Scenario
from telecom.base_station import BS
class LocationManager:
    """Handles agent location initialization and access."""
    
    def __init__(self, sce, scenario: Scenario, device: torch.device):
        self.sce = sce
        self.device = device
        self._location = self._initialize_location(scenario)
    
    def _initialize_location(self, scenario: Scenario) -> torch.Tensor:
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
    
    def __init__(self, sce, opt, index, device: torch.device):
        self.sce = sce
        self.opt = opt
        self.id = index
        self.device = device
        self.reward_scale = 1.0
    
    def compute(
        self,
        action_all: torch.Tensor,
        action_i: torch.Tensor,
        agent_location: torch.Tensor,
        scenario: Scenario,
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
            action_all, Ch_selected, BS_list, agent_location
        )

        
        sinr, capacity_mbps = self._compute_sinr_and_capacity(rx_power, interference)

        cap_tensor = torch.tensor(float(capacity_mbps), device=self.device)
        
        qos_thr_db = self.sce.QoS_thr
        sinr_db = 10.0 * np.log10(sinr + 1e-8)
        #print(f"action_i: {action_i}, rx_power: {rx_power}, interference: {interference}, sinr_db: {sinr_db}")

        delta_db = sinr_db - qos_thr_db

        shaped_reward = np.tanh(delta_db / 5.0)

        qos_satisfied = sinr_db >= qos_thr_db
        if qos_satisfied:
            shaped_reward = shaped_reward

        reward = shaped_reward * self.reward_scale

        return (
                int(qos_satisfied),
                torch.tensor(reward, device=self.device, dtype=torch.float32),
                cap_tensor,
            )
    
    def _invalid_action_result(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        scaled_negative = self.sce.negative_cost * self.reward_scale
        return (
            0,
            torch.tensor(scaled_negative, device=self.device, dtype=torch.float32),
            torch.tensor(0.0, device=self.device),
        )
    
    def _compute_rx_power(self, bs: BS, agent_location: torch.Tensor) -> float:
        BS_location = torch.tensor(bs.Get_Location(), device=self.device, dtype=torch.float32)
        Loc_diff = BS_location - agent_location
        distance = torch.sqrt(Loc_diff[0] ** 2 + Loc_diff[1] ** 2 + 1e-10)  
        d_val = max(float(distance.item()), 1.0)
        rx_power = bs.Receive_Power(d_val)
        if not np.isfinite(rx_power) or rx_power < 0:
            return 1e-20
        return rx_power
    
    def _compute_interference(
        self, action_all, channel, BS_list, agent_location
    ) -> float:
        K = self.sce.nChannel
        interference = 0.0

        for i in range(self.opt.nagents):
            if i == self.id:
                continue

            if int(action_all[i] % K) != channel:
                continue

            sel_bs_i = int(action_all[i] // K)
            if sel_bs_i >= len(BS_list):
                continue

            interference += self._compute_rx_power(
                BS_list[sel_bs_i],
                agent_location
            )

        return max(interference, 1e-20)
    
    def _compute_sinr_and_capacity(self, rx_power: float, interference: float) -> Tuple[float, float]:
        noise = 10 ** (self.sce.N0 / 10) * self.sce.BW
        denominator = max(interference + noise, 1e-20)
        sinr = rx_power / denominator
        if not np.isfinite(sinr) or sinr <= 0:
            return 0.0, 0.0
        spec_eff = np.log2(1.0 + sinr)
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
        self.max_grad_norm = 10.0
    
    def optimize(self, memory: ReplayMemory) -> Optional[NetworkMetrics]:
        """Performs a single optimization step using Double DQN."""
        if len(memory) < self.opt.batch_size:
            return None

        if (hasattr(self.opt, "min_memory_for_learning") 
            and len(memory) < self.opt.min_memory_for_learning):
            return None

        transitions = memory.Sample(self.opt.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        if (torch.isnan(state_batch).any() or torch.isnan(reward_batch).any() 
            or torch.isnan(next_state_batch).any()):
            return None
        
        all_q_values = self.model_policy(state_batch)
        q_pred = all_q_values.gather(1, action_batch)

        with torch.no_grad():
            next_actions = self.model_policy(next_state_batch).argmax(1, keepdim=True)
            q_next = self.model_target(next_state_batch).gather(1, next_actions)
            target_q = reward_batch.unsqueeze(1) + self.opt.gamma * q_next

        loss = F.smooth_l1_loss(q_pred, target_q)  
        if torch.isnan(loss):
            return  

        self.optimizer.zero_grad()
        loss.backward()

        grad_norm_before = compute_gradient_norm(self.model_policy)
        torch.nn.utils.clip_grad_norm_(self.model_policy.parameters(), max_norm=10.0)

        grad_clipped = grad_norm_before > self.max_grad_norm

        self.optimizer.step()

        metrics = NetworkMetrics(
            loss=float(loss.item()),
            mean_q=float(all_q_values.mean().item()),
            max_q=float(all_q_values.max().item()),
            min_q=float(all_q_values.min().item()),
            q_std=float(all_q_values.std().item()),
            grad_norm=grad_norm_before,
            grad_clipped=grad_clipped,
            target_q_mean=float(target_q.mean().item()),
            td_error=float((q_pred - target_q).abs().mean().item()), 
        )

        return metrics
    
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
    
    def select(self, q_values: torch.Tensor, epsilon: float, L: int, K: int) -> torch.Tensor:
        """Select action based on Q-values and exploration parameter."""
        if torch.isnan(q_values).any() or torch.isinf(q_values).any():
            return randrange(q_values.size(-1))
        
        if self.strategy == "epsilon_greedy":
            if random() < epsilon: 
                return q_values.argmax().item() 
            return randrange(L*K)
        
        elif self.strategy == "greedy":
            return int(q_values.argmax().item())
        
        elif self.strategy == "boltzmann":
            temperature = max(epsilon, 0.01)
            q_clipped = torch.clamp(q_values / temperature, min=-50, max=50)
            probs = torch.softmax(q_clipped, dim=-1)
            if torch.isnan(probs).any():
                return randrange(q_values.size(-1))
            return int(torch.multinomial(probs, 1).item())
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
class Agent:
    
    _all_locations: Optional[torch.Tensor] = None

    @classmethod
    def initialize_all_locations(cls, nagents: int, device: torch.device):
        cls._all_locations = torch.zeros(nagents, 2, device=device)

    @classmethod
    def get_all_locations(cls) -> Optional[torch.Tensor]:
        return cls._all_locations

    def __init__(self, opt, sce, scenario: Scenario, index: int, device: torch.device):
        self.opt = opt
        self.sce = sce
        self.id = index
        self.device = device
        
        # Composed components
        self._location_manager = LocationManager(sce, scenario, device)
        self._reward_calculator = RewardCalculator(sce, opt, index, device)
        self._action_selector = ActionSelector(
            strategy=getattr(opt, "action_strategy", "epsilon_greedy")
        )
        
        if Agent._all_locations is not None:
            Agent._all_locations[index] = self._location_manager.location

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

    def Select_Action(self, state: torch.Tensor, scenario: Scenario, eps: float) -> torch.Tensor:
        L = scenario.BS_Number()
        K = self.sce.nChannel
        q_values = self.model_policy(state)
        action = self._action_selector.select(
            q_values, eps, L, K
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

    def Optimize_Model(self) -> Optional[NetworkMetrics]:
        network_metrics = self._dqn_optimizer.optimize(self.memory)
        return network_metrics


__all__ = ["Agent", "ActionSelector", "RewardCalculator", "DQNOptimizer", "LocationManager"]