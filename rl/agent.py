"""Agent definition (UE) tying together network, memory, and learning logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import random, uniform, choice, randrange
from typing import Tuple, Optional, Dict, Union

import numpy as np
from numpy import pi
import torch

from constants import GRAD_CLIP_VALUE, USE_TENSOR_REPLAY_BUFFER
from telecom.mobility import MobilityManager


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

from .memory import ReplayMemory, TensorReplayMemory, Transition
from .networks import DNN

class LocationManager:
    """Handles agent location initialization and updates (with optional mobility)."""
    
    def __init__(
        self,
        sce,
        scenario,
        device: torch.device,
        ue_index: int = 0,
        mobility_manager: Optional[MobilityManager] = None,
    ):
        """Initialize location manager.
        
        Args:
            sce: Scenario configuration object.
            scenario: Scenario instance with BS locations.
            device: PyTorch device.
            ue_index: Index of this UE for mobility manager lookup.
            mobility_manager: Optional MobilityManager for position updates.
        """
        self.sce = sce
        self.scenario = scenario
        self.device = device
        self.ue_index = ue_index
        self._mobility_manager = mobility_manager
        
        # Initialize position - if mobility manager exists, it handles positions
        if self._mobility_manager is not None:
            # Let mobility manager provide position
            pos = self._mobility_manager.get_ue_position(ue_index)
            self._location = torch.tensor(pos, device=self.device, dtype=torch.float32)
        else:
            # Legacy initialization for static UEs
            self._location = self._initialize_location(scenario)
    
    def _initialize_location(self, scenario) -> torch.Tensor:
        """Initialize UE location within MBS coverage (legacy mode)."""
        Loc_MBS, _, _ = scenario.BS_Location()
        Loc_agent = np.zeros(2)
        LocM = choice(Loc_MBS)
        r = self.sce.rMBS * random()
        theta = uniform(-pi, pi)
        Loc_agent[0] = LocM[0] + r * np.cos(theta)
        Loc_agent[1] = LocM[1] + r * np.sin(theta)
        return torch.tensor(Loc_agent, device=self.device, dtype=torch.float32)
    
    def update_location(self) -> None:
        """Update location from mobility manager if available."""
        if self._mobility_manager is not None:
            pos = self._mobility_manager.get_ue_position(self.ue_index)
            self._location = torch.tensor(pos, device=self.device, dtype=torch.float32)
    
    @property
    def location(self) -> torch.Tensor:
        """Get current UE location as tensor."""
        return self._location
    
    @property
    def mobility_enabled(self) -> bool:
        """Check if mobility is enabled for this UE."""
        return self._mobility_manager is not None
    
class RewardCalculator:
    """Encapsulates reward computation logic with caching for performance."""
    
    def __init__(self, sce, opt, device: torch.device):
        self.sce = sce
        self.opt = opt
        self.device = device
        # Caches for performance - populated on first use
        self._bs_locations_np = None  # (nBS, 2) numpy array
        self._bs_locations_t = None   # (nBS, 2) torch tensor
        self._bs_tx_powers_t = None   # (nBS,) tx power in dBm
        self._bs_radii_t = None       # (nBS,) coverage radius
        self._bs_type_code_t = None   # (nBS,) 0 for MBS/PBS, 1 for FBS
        self._bs_list_cache = None
    
    def _init_bs_cache(self, BS_list):
        """Initialize BS location cache for vectorized operations."""
        if self._bs_list_cache is not BS_list or self._bs_locations_np is None:
            self._bs_list_cache = BS_list
            self._bs_locations_np = np.array([bs.Get_Location() for bs in BS_list], dtype=np.float32)
            self._bs_locations_t = torch.as_tensor(
                self._bs_locations_np, device=self.device, dtype=torch.float32
            )
            self._bs_tx_powers_t = torch.tensor(
                [float(bs.Transmit_Power_dBm()) for bs in BS_list],
                device=self.device,
                dtype=torch.float32,
            )
            self._bs_radii_t = torch.tensor(
                [float(getattr(bs, "radius", float("inf"))) for bs in BS_list],
                device=self.device,
                dtype=torch.float32,
            )
            self._bs_type_code_t = torch.tensor(
                [1.0 if getattr(bs, "bs_type", "MBS") == "FBS" else 0.0 for bs in BS_list],
                device=self.device,
                dtype=torch.float32,
            )

    def _rx_power_from_bs_index(self, bs_idx: int, agent_location: torch.Tensor) -> torch.Tensor:
        """Compute received power in mW using torch path-loss on the current device."""
        bs_loc = self._bs_locations_t[bs_idx]
        diff = bs_loc - agent_location
        d = torch.linalg.vector_norm(diff).clamp_min(1.0)

        tx_power_dbm = self._bs_tx_powers_t[bs_idx]
        is_fbs = self._bs_type_code_t[bs_idx]

        # MBS/PBS: 34 + 40log10(d), FBS: 37 + 30log10(d)
        loss_mbs_pbs = 34.0 + 40.0 * torch.log10(d)
        loss_fbs = 37.0 + 30.0 * torch.log10(d)
        loss = loss_mbs_pbs * (1.0 - is_fbs) + loss_fbs * is_fbs

        rx_power_dbm = tx_power_dbm - loss
        rx_mw = torch.pow(torch.tensor(10.0, device=self.device), rx_power_dbm / 10.0)

        # Keep behavior consistent with BS.receive_power: zero outside coverage radius.
        within_radius = d <= self._bs_radii_t[bs_idx]
        return torch.where(within_radius, rx_mw, torch.tensor(0.0, device=self.device))
    
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

        self._init_bs_cache(BS_list)

        rx_power_t = self._rx_power_from_bs_index(BS_selected, agent_location)
        rx_power = float(rx_power_t.detach().cpu())
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
        profit = float(getattr(self.sce, "profit", None) or 0.5)
        power_cost = float(getattr(self.sce, "power_cost", None) or 0.0005)
        action_cost = float(getattr(self.sce, "action_cost", None) or 0.001)
        
        # Get transmit power of selected BS
        tx_power_dbm = float(self._bs_tx_powers_t[BS_selected].detach().cpu())
        
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
        """Compute interference using vectorized torch operations on device."""
        K = self.sce.nChannel
        
        # Initialize BS cache if needed
        self._init_bs_cache(BS_list)
        
        # Keep actions on device to avoid host-device transfers.
        if isinstance(action_all, torch.Tensor):
            actions_t = action_all.to(self.device)
        else:
            actions_t = torch.as_tensor(action_all, device=self.device)
        actions_t = actions_t.to(torch.long).view(-1)
        
        # Vectorized channel and BS extraction
        channels = actions_t % K
        bs_indices = actions_t // K
        
        # Find agents on the same channel (excluding those with invalid BS)
        same_channel_mask = (channels == int(channel)) & (bs_indices >= 0) & (bs_indices < len(BS_list))
        
        if not bool(same_channel_mask.any()):
            return 0.0
        
        # Keep location on device
        if isinstance(agent_location, torch.Tensor):
            loc_t = agent_location.to(self.device, dtype=torch.float32)
        else:
            loc_t = torch.as_tensor(agent_location, device=self.device, dtype=torch.float32)
        
        # Get BS indices and locations for interfering agents
        interfering_bs_idx = bs_indices[same_channel_mask]
        interfering_bs_locs = self._bs_locations_t[interfering_bs_idx]
        
        # Vectorized distance calculation
        loc_diff = interfering_bs_locs - loc_t.unsqueeze(0)  # (n_interfering, 2)
        distances = torch.linalg.vector_norm(loc_diff, dim=1).clamp_min(1.0)

        # Per-BS pathloss model in torch (MBS/PBS vs FBS)
        tx_dbm = self._bs_tx_powers_t[interfering_bs_idx]
        is_fbs = self._bs_type_code_t[interfering_bs_idx]
        radii = self._bs_radii_t[interfering_bs_idx]

        loss_mbs_pbs = 34.0 + 40.0 * torch.log10(distances)
        loss_fbs = 37.0 + 30.0 * torch.log10(distances)
        loss = loss_mbs_pbs * (1.0 - is_fbs) + loss_fbs * is_fbs

        rx_dbm = tx_dbm - loss
        rx_mw = torch.pow(torch.tensor(10.0, device=self.device), rx_dbm / 10.0)
        rx_mw = torch.where(distances <= radii, rx_mw, torch.zeros_like(rx_mw))

        interference = float(rx_mw.sum().detach().cpu())
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
        from functions.gpu_manager import GPUManager
        _gm = GPUManager.from_opt(opt)
        self._amp_enabled = _gm.amp_enabled
        self._scaler = _gm.make_grad_scaler()
        self._metrics_interval = max(1, int(getattr(opt, "metrics_interval", 1) or 1))
        self._diag_interval = max(1, int(getattr(opt, "diag_interval", 1) or 1))
        self._opt_step = 0
        
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
    
    def optimize(self, memory: Union[ReplayMemory, 'TensorReplayMemory']) -> TrainMetrics:
        """Performs a single optimization step using Double DQN.
        
        Returns:
            TrainMetrics with statistics from this optimization step.
        """
        prev_metrics = self._last_metrics
        self._last_metrics = TrainMetrics()
        
        if len(memory) < self.opt.batch_size:
            return self._last_metrics

        if (hasattr(self.opt, "min_memory_for_learning") 
            and len(memory) < self.opt.min_memory_for_learning):
            return self._last_metrics

        sample_out = memory.Sample(self.opt.batch_size)

        if isinstance(sample_out, tuple) and len(sample_out) == 4 and torch.is_tensor(sample_out[0]):
            # TensorReplayMemory path: already batched tensors.
            state_batch, action_batch, next_state_batch, reward_batch = sample_out
            state_batch = state_batch.to(self.device)
            action_batch = action_batch.to(self.device)
            next_state_batch = next_state_batch.to(self.device)
            reward_batch = reward_batch.to(self.device)
        else:
            transitions = sample_out
            batch = Transition(*zip(*transitions))
            state_batch = torch.cat(batch.state).to(self.device)
            action_batch = torch.cat(batch.action).to(self.device)
            reward_batch = torch.cat(batch.reward).to(self.device)
            next_state_batch = torch.cat(batch.next_state).to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self._amp_enabled):
            q_all = self.model_policy(state_batch)
            q_pred = q_all.gather(1, action_batch)

            with torch.no_grad():
                next_actions = self.model_policy(next_state_batch).argmax(1, keepdim=True)
                q_next = self.model_target(next_state_batch).gather(1, next_actions)
                target_q = reward_batch.unsqueeze(1) + self.opt.gamma * q_next

            td_error = (q_pred - target_q).abs()
            loss = F.smooth_l1_loss(q_pred, target_q)

        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self.optimizer)

        self._opt_step += 1
        collect_diag = (self._opt_step % self._diag_interval) == 0
        collect_metrics = (self._opt_step % self._metrics_interval) == 0

        grad_norm = prev_metrics.grad_norm
        if collect_diag:
            grad_norm = self._compute_grad_norm()

        for param in self.model_policy.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-GRAD_CLIP_VALUE, GRAD_CLIP_VALUE)

        self._scaler.step(self.optimizer)
        self._scaler.update()

        loss_scalar = float(loss.detach().cpu())

        if collect_metrics:
            # Extract scalar diagnostics with one host transfer.
            metric_vec = torch.stack(
                [
                    q_all.mean().detach(),
                    q_all.max().detach(),
                    q_all.min().detach(),
                    q_all.std().detach(),
                    target_q.mean().detach(),
                    td_error.mean().detach(),
                ]
            ).cpu()
            mean_q = float(metric_vec[0])
            max_q = float(metric_vec[1])
            min_q = float(metric_vec[2])
            q_std = float(metric_vec[3])
            target_q_mean = float(metric_vec[4])
            td_error_value = float(metric_vec[5])
        else:
            mean_q = prev_metrics.mean_q
            max_q = prev_metrics.max_q
            min_q = prev_metrics.min_q
            q_std = prev_metrics.q_std
            target_q_mean = prev_metrics.target_q_mean
            td_error_value = prev_metrics.td_error

        # Store metrics
        self._last_metrics = TrainMetrics(
            loss=loss_scalar,
            mean_q=mean_q,
            max_q=max_q,
            min_q=min_q,
            q_std=q_std,
            grad_norm=grad_norm,
            target_q_mean=target_q_mean,
            td_error=td_error_value,
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
            "langevin": self._langevin,
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        return strategies[strategy]
    
    def _langevin(self, q_values: torch.Tensor, epsilon: float) -> int:
        """Langevin Monte Carlo action selection.
        
        Treats Q-values as an energy-based model and uses Langevin dynamics
        to sample actions. The epsilon parameter controls the temperature,
        with higher values leading to more exploration.
        
        Args:
            q_values: Q-values for all actions
            epsilon: Temperature parameter (higher = more exploration)
            
        Returns:
            Selected action index
        """
        temperature = max(epsilon, 0.05)
        step_size = 0.05
        n_steps = 15
        
        log_probs = q_values / temperature
        
        for _ in range(n_steps):
            probs = torch.softmax(log_probs, dim=-1)
            score = q_values / temperature - (probs * q_values).sum() / temperature
            noise = torch.randn_like(log_probs)
            log_probs = log_probs + (step_size / 2) * score + (step_size ** 0.5) * noise
        
        probs = torch.softmax(log_probs, dim=-1)
        return int(torch.multinomial(probs, 1).item())
    
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

    def select_batch(self, q_values_batch: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Batch action selection for tensor of shape (batch, n_actions)."""
        if q_values_batch.dim() != 2:
            raise ValueError("q_values_batch must have shape (batch, n_actions)")

        batch_size, n_actions = q_values_batch.shape
        device = q_values_batch.device

        if self.strategy == "greedy":
            return q_values_batch.argmax(dim=1)

        if self.strategy == "epsilon_greedy":
            greedy = q_values_batch.argmax(dim=1)
            random_actions = torch.randint(0, n_actions, (batch_size,), device=device)
            exploit_mask = torch.rand(batch_size, device=device) > epsilon
            return torch.where(exploit_mask, greedy, random_actions)

        if self.strategy == "epsilon_greedy_original":
            greedy = q_values_batch.argmax(dim=1)
            random_actions = torch.randint(0, n_actions, (batch_size,), device=device)
            exploit_mask = torch.rand(batch_size, device=device) < epsilon
            return torch.where(exploit_mask, greedy, random_actions)

        if self.strategy == "boltzmann":
            temperature = max(epsilon, 0.01)
            probs = torch.softmax(q_values_batch / temperature, dim=1)
            return torch.multinomial(probs, 1).squeeze(1)

        if self.strategy == "langevin":
            # Langevin uses iterative dynamics; keep per-row implementation.
            actions = [self._langevin(q_values_batch[i], epsilon) for i in range(batch_size)]
            return torch.tensor(actions, device=device, dtype=torch.long)

        raise ValueError(f"Unknown strategy: {self.strategy}")
        
class Agent:
    
    def __init__(
        self,
        opt,
        sce,
        scenario,
        index: int,
        device: torch.device,
        mobility_manager: Optional[MobilityManager] = None,
        input_dim: int = None,
        shared_networks: Optional[Tuple] = None,
        shared_memory: Optional['ReplayMemory'] = None,
    ):
        """Initialize an agent (UE) with optional mobility and GNN support.
        
        Args:
            opt: Optimization/training configuration.
            sce: Scenario configuration.
            scenario: Scenario instance.
            index: Agent/UE index.
            device: PyTorch device.
            mobility_manager: Optional MobilityManager for UE mobility.
            input_dim: Optional input dimension for GNN mode.
            shared_networks: Optional tuple of (model_policy, model_target, optimizer, dqn_optimizer)
                to share networks across agents. If None, creates independent networks.
            shared_memory: Optional shared ReplayMemory for all agents. If None, creates independent memory.
        """
        self.opt = opt
        self.sce = sce
        self.id = index
        self.device = device
        self._uses_shared_networks = shared_networks is not None
        
        # Composed components
        self._location_manager = LocationManager(
            sce,
            scenario,
            device,
            ue_index=index,
            mobility_manager=mobility_manager,
        )
        self._reward_calculator = RewardCalculator(sce, opt, device)
        self._action_selector = ActionSelector(
            strategy=(getattr(opt, "action_strategy", None) or "epsilon_greedy")
        )
        
        # Memory - shared or independent
        if shared_memory is not None:
            self.memory = shared_memory
        else:
            if USE_TENSOR_REPLAY_BUFFER:
                self.memory = TensorReplayMemory(opt.capacity)
            else:
                self.memory = ReplayMemory(opt.capacity)
        
        if shared_networks is not None:
            # Use shared networks
            self.model_policy, self.model_target, self.optimizer, self._dqn_optimizer = shared_networks
        else:
            # Create independent networks (original behavior)
            self.model_policy = DNN(opt, sce, scenario, input_dim=input_dim).to(device)
            self.model_target = DNN(opt, sce, scenario, input_dim=input_dim).to(device)
            self.model_target.load_state_dict(self.model_policy.state_dict())
            self.model_target.eval()
            
            # Optimizer component (with default momentum if not specified)
            momentum = opt.momentum if opt.momentum is not None else 0.0
            learning_rate = (
                getattr(opt, "learning_rate", None)
                or getattr(opt, "learningrate", None)
                or 5e-4
            )
            self.optimizer = optim.RMSprop(
                params=self.model_policy.parameters(),
                lr=learning_rate,
                momentum=momentum,
            )
            self._dqn_optimizer = DQNOptimizer(
                self.model_policy, self.model_target, self.optimizer, opt, device
            )
        
        # Cache for action space size
        self._n_actions = scenario.BS_Number() * sce.nChannel
    
    @property
    def uses_shared_networks(self) -> bool:
        """Check if this agent uses shared networks."""
        return self._uses_shared_networks

    @property
    def location(self) -> torch.Tensor:
        return self._location_manager.location

    def Get_Location(self) -> torch.Tensor:
        return self.location
    
    def update_location(self) -> None:
        """Update UE location from mobility manager (called at episode start)."""
        self._location_manager.update_location()
    
    @property
    def mobility_enabled(self) -> bool:
        """Check if mobility is enabled for this UE."""
        return self._location_manager.mobility_enabled

    def Select_Action(self, state: torch.Tensor, scenario, eps: float) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.model_policy(state)
        
        action = self._action_selector.select(
            q_values, eps
        )

        return torch.tensor([[action]], device=self.device, dtype=torch.long)
    
    def get_q_diagnostics(self, state: torch.Tensor) -> dict:
        """Get Q-value diagnostics for debugging action selection.
        
        Returns:
            dict with Q-value statistics to diagnose if the network
            is properly differentiating between good and bad actions.
        """
        with torch.no_grad():
            q_values = self.model_policy(state)
            q_flat = q_values.flatten()
            
            best_action = int(q_flat.argmax().item())
            worst_action = int(q_flat.argmin().item())
            
            # Q-value spread indicates how much the network differentiates actions
            q_spread = float(q_flat.max().item() - q_flat.min().item())
            q_std = float(q_flat.std().item())
            
            # Top-k actions
            k = min(5, q_flat.size(0))
            top_k_values, top_k_indices = torch.topk(q_flat, k)
            
            return {
                "best_action": best_action,
                "best_q": float(q_flat.max().item()),
                "worst_action": worst_action,
                "worst_q": float(q_flat.min().item()),
                "mean_q": float(q_flat.mean().item()),
                "q_spread": q_spread,
                "q_std": q_std,
                "top_k_actions": top_k_indices.tolist(),
                "top_k_q_values": top_k_values.tolist(),
            }

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