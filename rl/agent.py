"""Agent definition (UE) tying together network, memory, and learning logic."""

from __future__ import annotations

from .optimization import DQNOptimizer, TrainMetrics
from telecom.hetnet_reward import RewardCalculator

from random import random, uniform, choice, randrange
from typing import Tuple, Optional, Dict, Union

import numpy as np
from numpy import pi
import torch

from functions.configuration import config_value
from constants import USE_TENSOR_REPLAY_BUFFER
from telecom.environment_factory import create_reward_calculator
from telecom.mobility import MobilityManager


# Optimizer construction
from torch import optim

from .memory import ReplayMemory, TensorReplayMemory
from .networks import build_model

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
        """Initialize UE location. Uses area-uniform placement for cell-free,
        or MBS-anchored placement for HetNet (legacy mode)."""
        Loc_agent = np.zeros(2)
        rMBS = getattr(self.sce, "rMBS", None)
        if rMBS is not None:
            # HetNet: place within a random MBS coverage radius
            Loc_MBS, _, _ = scenario.BS_Location()
            LocM = scenario.rng.choice(Loc_MBS)
            r = rMBS * scenario.rng.random()
            theta = scenario.rng.uniform(-pi, pi)
            Loc_agent[0] = LocM[0] + r * np.cos(theta)
            Loc_agent[1] = LocM[1] + r * np.sin(theta)
        else:
            # Cell-Free: place uniformly within the deployment area
            area_w = float(getattr(self.sce, "area_width", 2000.0) or 2000.0)
            area_h = float(getattr(self.sce, "area_height", area_w) or area_w)
            Loc_agent[0] = uniform(0.0, area_w)
            Loc_agent[1] = uniform(0.0, area_h)
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
        if not torch.isfinite(q_values).all():
            return int(torch.nan_to_num(q_values, nan=float("-inf")).argmax().item())

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
        if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
            return int(q_values.argmax().item())
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
        if not torch.isfinite(probs).all() or float(probs.sum().item()) <= 0.0:
            return int(q_values.argmax().item())
        return int(torch.multinomial(probs, 1).item())
    
    def select(self, q_values: torch.Tensor, epsilon: float) -> int:
        """Select action based on Q-values and exploration parameter.
        
        Args:
            q_values: Q-values for all actions
            epsilon: Exploration threshold 
        Returns:
            Selected action index
        """
        if not torch.isfinite(q_values).all():
            raise FloatingPointError("Non-finite action values")
        return self._select(q_values, epsilon)

    def select_batch(self, q_values_batch: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Batch action selection for tensor of shape (batch, n_actions)."""
        if not torch.isfinite(q_values_batch).all():
            raise FloatingPointError("Non-finite action values")
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
        self._reward_calculator = create_reward_calculator(
            sce,
            opt,
            device,
            fallback_cls=RewardCalculator,
        )
        self._action_selector = ActionSelector(
            strategy=(getattr(opt, "action_strategy", None) or "epsilon_greedy")
        )
        
        # Memory - shared or independent
        if shared_memory is not None:
            self.memory = shared_memory
        else:
            if USE_TENSOR_REPLAY_BUFFER:
                self.memory = TensorReplayMemory(opt.capacity, sampling=config_value(opt, "replay_sampling", "legacy"))
            else:
                self.memory = ReplayMemory(opt.capacity, sampling=config_value(opt, "replay_sampling", "legacy"))
        
        if shared_networks is not None:
            # Use shared networks
            self.model_policy, self.model_target, self.optimizer, self._dqn_optimizer = shared_networks
        else:
            # Create independent networks (original behavior)
            self.model_policy = build_model(opt, sce, scenario, input_dim=input_dim).to(device)
            self.model_target = build_model(opt, sce, scenario, input_dim=input_dim).to(device)
            self.model_target.load_state_dict(self.model_policy.state_dict())
            self.model_target.eval()
            
            # Optimizer component (with default momentum if not specified)
            momentum = config_value(opt, "momentum", 0.0)
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
    
    def set_location(self, position):
        self._location_manager._location = torch.as_tensor(position, device=self.device, dtype=torch.float32).clone()

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
            q_std = float(q_flat.std().item()) if q_flat.numel() > 1 else 0.0
            
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
        kwargs = {"ue_index": self._location_manager.ue_index} if hasattr(scenario, "begin_step") else {}
        return self._reward_calculator.compute(
            action_all, action_i, self.location, scenario, **kwargs
        )

    def load_policy_state(self, state_dict: Dict[str, torch.Tensor]):
        """Load a saved policy model state dict without reinitializing the optimizer."""
        self.model_policy.load_state_dict(state_dict)
        self.model_target.load_state_dict(state_dict)

    def set_eval_mode(self):
        """Set the policy and target networks to evaluation mode for inference-only runs."""
        self.model_policy.eval()
        self.model_target.eval()

    def set_train_mode(self):
        """Set the policy and target networks to training mode for standard updates."""
        self.model_policy.train()
        self.model_target.train()

    def Save_Transition(self, state, action, next_state, reward, scenario, terminated=False):
        self.memory.Push(
            state.unsqueeze(0),
            action.view(1, 1),
            next_state.unsqueeze(0),
            reward.view(1),
            terminated,
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