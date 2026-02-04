"""Agent definition (UE) tying together network, memory, and learning logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import random, uniform, choice, randrange
from typing import Tuple, Optional, Dict

import numpy as np
from numpy import pi
import torch

from constants import GRAD_CLIP_VALUE


@dataclass
class TrainMetrics:
    """Metrics from a single DQN optimization step."""
    loss: float = 0.0
    mean_q: float = 0.0
    max_q: float = 0.0
    min_q: float = 0.0
    q_std: float = 0.0
    grad_norm: float = 0.0
    target_q_mean: float = 0.0
    td_error: float = 0.0
    did_learn: bool = False  # Whether learning actually happened

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "loss": self.loss,
            "mean_q": self.mean_q,
            "max_q": self.max_q,
            "min_q": self.min_q,
            "q_std": self.q_std,
            "grad_norm": self.grad_norm,
            "target_q_mean": self.target_q_mean,
            "td_error": self.td_error,
        }

# Import Adam optimizer
from torch import optim
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
        """Compute QoS satisfaction, reward, and capacity.
        
        Reward = profit * Rate - power_cost * Tx_power_dBm - action_cost
        (utility-based reward with action selection cost)
        """
        BS_list = scenario.Get_BaseStations()
        K = self.sce.nChannel

        BS_selected = int(action_i // K)
        Ch_selected = int(action_i % K)

        # Basic validation for action index
        if BS_selected >= len(BS_list):
            return self._invalid_action_result()

        selected_bs = BS_list[BS_selected]
        rx_power = self._compute_rx_power(selected_bs, agent_location)
        if rx_power <= 1e-20:
            return self._invalid_action_result()

        interference = self._compute_interference(
            action_all, Ch_selected, BS_list, agent_location, rx_power
        )
        
        sinr, capacity_mbps = self._compute_sinr_and_capacity(rx_power, interference)
        cap_tensor = torch.tensor(float(capacity_mbps), device=self.device)

        # Compute utility-based reward
        # Rate in Mbps (same as capacity_mbps)
        rate = capacity_mbps
        
        # Get config parameters with defaults
        profit = float(getattr(self.sce, "profit", 0.5))
        power_cost = float(getattr(self.sce, "power_cost", 0.0005))
        action_cost = float(getattr(self.sce, "action_cost", 0.001))
        
        # Get transmit power of selected BS
        tx_power_dbm = selected_bs.Transmit_Power_dBm()
        
        # Calculate reward: utility (profit * rate) minus costs
        reward_val = profit * rate - power_cost * tx_power_dbm - action_cost
        
        # Determine QoS satisfaction (binary)
        if sinr >= 10 ** (self.sce.QoS_thr / 10):
            qos = 1
        else:
            qos = 0

        return (qos, torch.tensor(reward_val, device=self.device, dtype=torch.float32), cap_tensor)
    
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
        
        # Store last training metrics
        self._last_metrics: TrainMetrics = TrainMetrics()
    
    @property
    def last_metrics(self) -> TrainMetrics:
        """Get metrics from the last optimization step."""
        return self._last_metrics
    
    def _compute_grad_norm(self) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for param in self.model_policy.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def optimize(self, memory: ReplayMemory) -> TrainMetrics:
        """Performs a single optimization step using Double DQN.
        
        Returns:
            TrainMetrics with statistics from this optimization step.
        """
        self._last_metrics = TrainMetrics()
        
        if len(memory) < self.opt.batch_size:
            return self._last_metrics

        if (hasattr(self.opt, "min_memory_for_learning") 
            and len(memory) < self.opt.min_memory_for_learning):
            return self._last_metrics

        transitions = memory.Sample(self.opt.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)

        q_all = self.model_policy(state_batch)
        q_pred = q_all.gather(1, action_batch)

        with torch.no_grad():
            next_actions = self.model_policy(next_state_batch).argmax(1, keepdim=True)
            q_next = self.model_target(next_state_batch).gather(1, next_actions)
            target_q = reward_batch.unsqueeze(1) + self.opt.gamma * q_next

        td_error = (q_pred - target_q).abs()
        
        loss = F.smooth_l1_loss(q_pred, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norm = self._compute_grad_norm()
        
        for param in self.model_policy.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE)
        self.optimizer.step()
        
        # Store metrics
        self._last_metrics = TrainMetrics(
            loss=loss.item(),
            mean_q=q_all.mean().item(),
            max_q=q_all.max().item(),
            min_q=q_all.min().item(),
            q_std=q_all.std().item(),
            grad_norm=grad_norm,
            target_q_mean=target_q.mean().item(),
            td_error=td_error.mean().item(),
            did_learn=True,
        )
        
        return self._last_metrics
    
    def soft_update(self, tau: float = 0.005):
        """Performs soft update of the target network parameters."""
        target_state_dict = self.model_target.state_dict()
        policy_state_dict = self.model_policy.state_dict()
        for key in policy_state_dict:
            target_state_dict[key] = (
                policy_state_dict[key] * tau + target_state_dict[key] * (1.0 - tau)
            )
        self.model_target.load_state_dict(target_state_dict)
    
    def hard_update(self):
        """Performs hard update of target network (full copy) like original UARA-DRL."""
        self.model_target.load_state_dict(self.model_policy.state_dict())

class ActionSelector:
    """Encapsulates action selection strategy (epsilon-greedy, Boltzmann, etc.).
    """
    
    def __init__(self, strategy: str = "epsilon_greedy"):
        self.strategy = strategy
        self._select = self._get_strategy_fn(strategy)
    
    def _get_strategy_fn(self, strategy: str):
        """Return the selection function for the given strategy."""
        strategies = {
            "epsilon_greedy": self._epsilon_greedy,
            "epsilon_greedy_original": self._epsilon_greedy_original,
            "greedy": self._greedy,
            "boltzmann": self._boltzmann,
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        return strategies[strategy]
    
    def _epsilon_greedy_original(self, q_values: torch.Tensor, epsilon: float) -> int:
        if random() < epsilon:
            return int(q_values.argmax().item())
        return randrange(q_values.size(-1))
    
    def _epsilon_greedy(self, q_values: torch.Tensor, epsilon: float) -> int:
        if random() > epsilon:
            return int(q_values.argmax().item())
        return randrange(q_values.size(-1))
    
    def _greedy(self, q_values: torch.Tensor, epsilon: float) -> int:
        return int(q_values.argmax().item())
    
    def _boltzmann(self, q_values: torch.Tensor, epsilon: float) -> int:
        temperature = max(epsilon, 0.01)  
        probs = torch.softmax(q_values / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())
    
    def select(self, q_values: torch.Tensor, epsilon: float) -> int:
        """Select action based on Q-values and exploration parameter.
        
        Args:
            q_values: Q-values for all actions
            epsilon: Exploration threshold 
        Returns:
            Selected action index
        """
        return self._select(q_values, epsilon)
        
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
        self.optimizer = optim.RMSprop(
            params=self.model_policy.parameters(),
            lr=opt.learning_rate,
            momentum=opt.momentum,
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
    
    def Target_Update(self):
        """Hard update of target network (full copy) like original UARA-DRL."""
        self._dqn_optimizer.hard_update()

    def Optimize_Model(self) -> TrainMetrics:
        """Optimize the policy network and return training metrics."""
        return self._dqn_optimizer.optimize(self.memory)
    
    def get_train_metrics(self) -> TrainMetrics:
        """Get metrics from the last optimization step."""
        return self._dqn_optimizer.last_metrics


__all__ = ["Agent", "ActionSelector", "RewardCalculator", "DQNOptimizer", "LocationManager", "TrainMetrics"]