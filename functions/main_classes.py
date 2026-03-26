import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import torch

from telecom.scenario import Scenario
from telecom.mobility import MobilityManager, MOBILITY_ENABLED
from telecom.interference import compute_interference_graph
from rl.training import (
    select_actions,
    compute_rewards_and_next_state,
    store_and_learn,
    create_agents,
    initialize_episode,
    should_terminate,
    GNNObservationManager,
)
from functions.live_plot import RealTimeStepPlotter
from functions.network_metrics_plot import NetworkMetricsPlotter
from functions.telecom_network_plot import TelecomNetworkPlotter
from functions.logging import (
    RunDirectoryManager,
    save_config,
    save_environment_snapshot,
    EpisodeMetricsLogger,
    StepMetricsLogger,
    checkpoint_agents,
    write_summary,
    write_experiment_summary,
)
from constants import (
    NETWORK_PLOT_ENABLED,
    NETWORK_PLOT_INTERVAL,
    NETWORK_PLOT_SMOOTH_WINDOW,
    NETWORK_PLOT_FILENAME,
    EPS_MAX_DECAY,
    EPS_MAX_MIN,
    TELECOM_PLOT_ENABLED,
    TELECOM_PLOT_INTERVAL,
    TELECOM_PLOT_FILENAME,
    GNN_ENABLED,
)


@dataclass
class TrainingConfig:
    """Typed configuration with defaults for training parameters."""
    nepisodes: int
    nsteps: int
    nagents: int
    enable_plot: bool = True
    plot_interval: int = 10
    plot_smooth_window: int = 50
    plot_x_axis: str = "episodes"
    early_termination: bool = True
    step_log_throttle: int = 1
    checkpoint_interval: int = 100
    # Epsilon schedule
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    
    @classmethod
    def from_opt(cls, opt) -> "TrainingConfig":
        """Create config from DotDic/namespace object with defaults."""
        return cls(
            nepisodes=opt.nepisodes,
            nsteps=opt.nsteps,
            nagents=opt.nagents,
            enable_plot=getattr(opt, "enable_plot", True),
            plot_interval=getattr(opt, "plot_interval", 10),
            plot_smooth_window=getattr(opt, "plot_smooth_window", 50),
            plot_x_axis=getattr(opt, "plot_x_axis", "episodes"),
            early_termination=getattr(opt, "early_termination", True),
            step_log_throttle=getattr(opt, "step_log_throttle", 1),
            checkpoint_interval=getattr(opt, "checkpoint_interval", 100),
        )


class EpsilonScheduler:
    """Handles epsilon scheduling for exploration.
    
    Supports two policies:
    - 'exponential_decay': Standard RL approach, starts high and decays
    - 'linear_increasing': Like original UARA-DRL, starts low and increases
    """
    
    def __init__(self, policy: str, eps_min: float, eps_max: float, 
                 eps_increment: float, eps_start: float, eps_end: float, 
                 eps_decay_steps: int, nepisodes: int, nsteps: int,
                 eps_max_decay: float = 1.0, eps_max_min: float = 0.1):
        self.policy = policy
        self.eps_min = eps_min
        self.eps_max_initial = eps_max 
        self.eps_max = eps_max  
        self.eps_increment = eps_increment
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.nepisodes = nepisodes
        self.nsteps = nsteps
        self.current_episode = 0
        self.current_step_in_episode = 0        
        self.eps_max_decay = eps_max_decay 
        self.eps_max_min = eps_max_min  
    
    def set_episode(self, episode: int):
        """Set current episode and update decaying eps_max."""
        self.current_episode = episode
        self.current_step_in_episode = 0
        
        if self.eps_max_decay < 1.0:
            self.eps_max = max(
                self.eps_max_min,
                self.eps_max_initial * (self.eps_max_decay ** episode)
            )
    
    def get_epsilon(self, current_step: int) -> float:
        """Compute epsilon based on selected policy."""
        if self.policy == "linear_increasing":
            eps = self.eps_min + self.eps_increment * self.current_step_in_episode * (self.current_episode + 1)
            if eps > self.eps_max:
                eps = self.eps_max
            self.current_step_in_episode += 1
            return eps
        else:
            if self.eps_decay_steps <= 0:
                return self.eps_end
            epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
                -1.0 * current_step / self.eps_decay_steps
            )
            return epsilon
    
    @classmethod
    def from_opt(cls, opt) -> "EpsilonScheduler":
        """Create scheduler from opt DotDic object."""
        # Helper to handle None values from DotDic
        def get_or_default(name, default):
            val = getattr(opt, name, None)
            return default if val is None else val
        
        return cls(
            policy=get_or_default("epsilon_policy", "exponential_decay"),
            eps_min=get_or_default("eps_min", 0.0),
            eps_max=get_or_default("eps_max", 0.9),
            eps_increment=get_or_default("eps_increment", 0.003),
            eps_start=get_or_default("eps_start", 1.0),
            eps_end=get_or_default("eps_end", 0.05),
            eps_decay_steps=get_or_default("eps_decay_steps", 10000),
            nepisodes=get_or_default("nepisodes", 1000),
            nsteps=get_or_default("nsteps", 500),
            eps_max_decay=EPS_MAX_DECAY,
            eps_max_min=EPS_MAX_MIN,
        )


@dataclass
class EpisodeMetrics:
    """Accumulates metrics within a single episode for diagnosis.
    
    Tracks both environment metrics (rewards, QoS, capacity) and
    learning diagnostics (loss, Q-values, gradients).
    """
    reward_sum: float = 0.0
    return_total: float = 0.0
    qos_sum: float = 0.0
    capacity_sum: float = 0.0
    steps: int = 0
    
    # Track reward distribution
    reward_min: float = float('inf')
    reward_max: float = float('-inf')
    reward_squared_sum: float = 0.0  # For std calculation
    
    # Learning diagnostics accumulators
    loss_sum: float = 0.0
    mean_q_sum: float = 0.0
    max_q_sum: float = 0.0
    td_error_sum: float = 0.0
    grad_norm_sum: float = 0.0
    learning_steps: int = 0  # Steps where actual learning occurred
    memory_size: int = 0  # Replay buffer size at episode end
    
    # QoS success tracking
    qos_success_count: int = 0  # Steps where QoS >= threshold
    
    def record_step(self, rewards: torch.Tensor, qos: torch.Tensor, capacity: torch.Tensor):
        """Record environment metrics from a single step."""
        mean_reward = float(rewards.mean().item())
        self.reward_sum += mean_reward
        self.return_total += float(rewards.sum().item())
        self.qos_sum += float(qos.mean().item())
        self.capacity_sum += float(capacity.sum().item())
        
        # Track reward distribution
        self.reward_min = min(self.reward_min, mean_reward)
        self.reward_max = max(self.reward_max, mean_reward)
        self.reward_squared_sum += mean_reward ** 2
        
        # Track QoS success (full satisfaction)
        if float(qos.mean().item()) >= 1.0:
            self.qos_success_count += 1
        
        self.steps += 1
    
    def record_learning(self, train_metrics: Dict[str, float]):
        """Record learning diagnostics from a training step."""
        if train_metrics is None:
            return
        
        self.loss_sum += train_metrics.get("loss", 0.0)
        self.mean_q_sum += train_metrics.get("mean_q", 0.0)
        self.max_q_sum += train_metrics.get("max_q", 0.0)
        self.td_error_sum += train_metrics.get("td_error", 0.0)
        self.grad_norm_sum += train_metrics.get("grad_norm", 0.0)
        self.learning_steps += 1
    
    def set_memory_size(self, size: int):
        """Set replay buffer size at episode end."""
        self.memory_size = size
    
    @property
    def avg_reward(self) -> float:
        return self.reward_sum / max(1, self.steps)
    
    @property
    def reward_std(self) -> float:
        """Standard deviation of rewards in this episode."""
        if self.steps < 2:
            return 0.0
        mean = self.avg_reward
        variance = (self.reward_squared_sum / self.steps) - (mean ** 2)
        return max(0.0, variance) ** 0.5
    
    @property
    def qos_rate(self) -> float:
        return self.qos_sum / max(1, self.steps)
    
    @property
    def avg_capacity(self) -> float:
        return self.capacity_sum / max(1, self.steps)
    
    @property
    def success_rate(self) -> float:
        """Fraction of steps with full QoS satisfaction."""
        return self.qos_success_count / max(1, self.steps)
    
    # Learning diagnostics averages
    @property
    def loss_avg(self) -> float:
        return self.loss_sum / max(1, self.learning_steps)
    
    @property
    def mean_q_avg(self) -> float:
        return self.mean_q_sum / max(1, self.learning_steps)
    
    @property
    def max_q_avg(self) -> float:
        return self.max_q_sum / max(1, self.learning_steps)
    
    @property
    def td_error_avg(self) -> float:
        return self.td_error_sum / max(1, self.learning_steps)
    
    @property
    def grad_norm_avg(self) -> float:
        return self.grad_norm_sum / max(1, self.learning_steps)


class MetricsAggregator:
    """Aggregates metrics across all episodes."""
    
    def __init__(self, convergence_window: int = 50, convergence_threshold: float = 0.01):
        self.avg_rewards: List[float] = []
        self.returns: List[float] = []
        self.qos_rates: List[float] = []
        
        # Best episode tracking
        self.best_episode: Optional[Dict[str, Any]] = None
        self.best_return: float = float('-inf')
        
        # Convergence tracking
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold
        self.convergence_episode: Optional[int] = None
        
        # Learning diagnostics tracking
        self.losses: List[float] = []
        self.mean_qs: List[float] = []
        self.td_errors: List[float] = []
        self.grad_norms: List[float] = []
    
    def record_episode(self, metrics: EpisodeMetrics):
        """Record metrics from a completed episode."""
        self.avg_rewards.append(metrics.avg_reward)
        self.returns.append(metrics.return_total)
        self.qos_rates.append(metrics.qos_rate)
        
        # Track best episode (by return)
        episode_idx = len(self.returns) - 1
        if metrics.return_total > self.best_return:
            self.best_return = metrics.return_total
            self.best_episode = {
                "episode": episode_idx,
                "return": metrics.return_total,
                "avg_reward": metrics.avg_reward,
                "qos_rate": metrics.qos_rate,
                "steps": metrics.steps,
            }
        
        # Check for convergence (return stabilization)
        self._check_convergence(episode_idx)
    
    def record_learning_step(self, loss: Optional[float] = None, 
                             mean_q: Optional[float] = None,
                             td_error: Optional[float] = None,
                             grad_norm: Optional[float] = None):
        """Record learning diagnostics from a training step."""
        if loss is not None:
            self.losses.append(loss)
        if mean_q is not None:
            self.mean_qs.append(mean_q)
        if td_error is not None:
            self.td_errors.append(td_error)
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
    
    def _check_convergence(self, current_episode: int):
        """Check if training has converged based on return stability."""
        if self.convergence_episode is not None:
            return  # Already converged
        
        if len(self.returns) < self.convergence_window * 2:
            return  # Not enough data
        
        # Compare recent window to previous window
        recent = self.returns[-self.convergence_window:]
        previous = self.returns[-2*self.convergence_window:-self.convergence_window]
        
        recent_mean = sum(recent) / len(recent)
        previous_mean = sum(previous) / len(previous)
        
        # Check if relative change is below threshold
        if previous_mean != 0:
            relative_change = abs(recent_mean - previous_mean) / abs(previous_mean)
            if relative_change < self.convergence_threshold:
                self.convergence_episode = current_episode - self.convergence_window
    
    def mean_last_k(self, data: List[float], k: int = 100) -> Optional[float]:
        """Compute mean of the last k values."""
        if not data:
            return None
        return float(sum(data[-k:]) / max(1, len(data[-k:])))
    
    def get_summary(self, k: int = 100) -> Dict[str, Optional[float]]:
        """Generate summary statistics."""
        return {
            "mean_last_100_reward": self.mean_last_k(self.avg_rewards, k),
            "mean_last_100_return": self.mean_last_k(self.returns, k),
            "mean_last_100_qos": self.mean_last_k(self.qos_rates, k),
        }
    
    def get_final_metrics(self, k: int = 100) -> Dict[str, Any]:
        """Get comprehensive final metrics for experiment summary."""
        return {
            "mean_reward": self.mean_last_k(self.avg_rewards, k),
            "mean_return": self.mean_last_k(self.returns, k),
            "mean_qos": self.mean_last_k(self.qos_rates, k),
            "std_reward": self._std_last_k(self.avg_rewards, k),
            "std_return": self._std_last_k(self.returns, k),
            "std_qos": self._std_last_k(self.qos_rates, k),
            "max_return": max(self.returns) if self.returns else None,
            "min_return": min(self.returns) if self.returns else None,
        }
    
    def get_best_episode(self) -> Optional[Dict[str, Any]]:
        """Get information about the best performing episode."""
        return self.best_episode
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get convergence detection information."""
        return {
            "converged": self.convergence_episode is not None,
            "convergence_episode": self.convergence_episode,
            "convergence_window": self.convergence_window,
            "convergence_threshold": self.convergence_threshold,
        }
    
    def get_learning_diagnostics(self, k: int = 100) -> Dict[str, Any]:
        """Get learning diagnostics summary."""
        return {
            "final_loss": self.mean_last_k(self.losses, k),
            "final_mean_q": self.mean_last_k(self.mean_qs, k),
            "final_td_error": self.mean_last_k(self.td_errors, k),
            "final_grad_norm": self.mean_last_k(self.grad_norms, k),
            "total_learning_steps": len(self.losses),
        }
    
    def _std_last_k(self, data: List[float], k: int = 100) -> Optional[float]:
        """Compute standard deviation of the last k values."""
        if not data or len(data) < 2:
            return None
        subset = data[-k:]
        mean = sum(subset) / len(subset)
        variance = sum((x - mean) ** 2 for x in subset) / len(subset)
        return float(variance ** 0.5)
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Export all metrics as a dictionary."""
        return {
            "avg_reward": self.avg_rewards,
            "return": self.returns,
            "qos_rate": self.qos_rates,
        }


class EpisodeRunner:
    """Handles execution of a single episode."""
    
    def __init__(
        self,
        agents: List,
        scenario: Scenario,
        config: TrainingConfig,
        device: torch.device,
        action_tail_len: int = 10,
        gnn_manager = None,
    ):
        self.agents = agents
        self.scenario = scenario
        self.config = config
        self.device = device
        self.action_tail_len = action_tail_len
        self.state_target = torch.ones(config.nagents, device=device)
        self.gnn_manager = gnn_manager
        
        # Episode state (reset each episode)
        self.ctx = None
        self.metrics = None
        self.last_actions = None
        self.step = 0
        self.last_epsilon = 0.0
        self._terminated_early = False
    
    def reset(self):
        """Initialize state for a new episode."""
        self.ctx = initialize_episode(self.config.nagents, self.device)
        self.metrics = EpisodeMetrics()
        self.last_actions = [
            deque(maxlen=self.action_tail_len) 
            for _ in range(self.config.nagents)
        ]
        self.step = 0
        self._terminated_early = False
        
        # GNN observation storage (current and next)
        self._gnn_obs = None
        self._gnn_obs_next = None
    
    def run_step(self, epsilon: float) -> Dict[str, float]:
        """Execute a single step and return step metrics."""
        self.last_epsilon = epsilon
        
        # Get GNN observations if enabled
        gnn_obs = None
        if self.gnn_manager and self.gnn_manager.enabled:
            gnn_obs = self.gnn_manager.encode(
                self.agents, 
                self.ctx.actions if self.step > 0 else None
            )
            self._gnn_obs = gnn_obs
        
        # Select actions (with GNN observations if available)
        self.ctx.actions = select_actions(
            self.agents, self.ctx.state, self.scenario, epsilon,
            gnn_observations=gnn_obs,
        )
        
        # Record actions for debugging
        self._record_actions()
        
        # Environment step
        self.ctx.rewards, self.ctx.qos, self.ctx.next_state, capacity = (
            compute_rewards_and_next_state(
                self.agents, self.ctx.actions, self.ctx.state, self.scenario
            )
        )
        
        # Record metrics
        self.metrics.record_step(self.ctx.rewards, self.ctx.qos, capacity)
        
        # Prepare step metrics for logging
        step_metrics = {
            "mean_reward": float(self.ctx.rewards.mean().item()),
            "qos_mean": float(self.ctx.qos.mean().item()),
            "capacity_sum": float(capacity.sum().item()),
        }
        
        return step_metrics
    
    def learn(self, opt) -> Optional[Dict[str, float]]:
        """Store experience and update networks.
        
        Returns:
            Aggregated training metrics if learning occurred, None otherwise.
        """
        # Get next GNN observations for storing transitions
        gnn_obs_next = None
        if self.gnn_manager and self.gnn_manager.enabled:
            gnn_obs_next = self.gnn_manager.encode(
                self.agents,
                self.ctx.actions,  # Use current actions for next state graph
            )
        
        return store_and_learn(
            self.agents,
            self.ctx.state,
            self.ctx.actions,
            self.ctx.next_state,
            self.ctx.rewards,
            self.scenario,
            self.step,
            opt,
            gnn_obs=self._gnn_obs,
            gnn_obs_next=gnn_obs_next,
        )
    
    def advance(self):
        """Transition to next state."""
        self.ctx.state = self.ctx.next_state.clone()
        if self.config.early_termination:
            if should_terminate(self.ctx.state, self.state_target):
                self._terminated_early = True
        self.step += 1
    
    def is_done(self) -> bool:
        """Check if episode should terminate."""
        if self._terminated_early or self.step >= self.config.nsteps:
            return True
        if self.config.early_termination and self.step > 0:
            return should_terminate(self.ctx.state, self.state_target)
        return False
    
    def _record_actions(self):
        """Record actions for debugging."""
        if self.action_tail_len > 0:
            for i in range(self.config.nagents):
                try:
                    self.last_actions[i].append(int(self.ctx.actions[i].item()))
                except Exception:
                    self.last_actions[i].append(int(self.ctx.actions[i]))


class Trainer:
    """Orchestrates the training loop across episodes."""
    
    def __init__(
        self,
        agents: List,
        scenario: Scenario,
        config: TrainingConfig,
        opt: Any,  # Original opt for store_and_learn compatibility
        sce: Any,  # Scenario config for summary generation
        run_dir: RunDirectoryManager,
        device: torch.device,
        mobility_manager: Optional[MobilityManager] = None,
        gnn_manager: GNNObservationManager = None,
    ):
        self.agents = agents
        self.scenario = scenario
        self.config = config
        self.opt = opt
        self.sce = sce
        self.run_dir = run_dir
        self.device = device
        self.mobility_manager = mobility_manager
        self.gnn_manager = gnn_manager
        
        # Components
        self.epsilon_scheduler = EpsilonScheduler.from_opt(opt)
        self.episode_runner = EpisodeRunner(agents, scenario, config, device, gnn_manager=gnn_manager)
        self.metrics_aggregator = MetricsAggregator()
        
        self.plotter = self._create_plotter()
        self.network_plotter = self._create_network_plotter()
        self.telecom_plotter = self._create_telecom_plotter()
        self.ep_logger = EpisodeMetricsLogger(run_dir)
        self.step_logger = StepMetricsLogger(run_dir, throttle=config.step_log_throttle)
        
        # Training state
        self.total_steps = 0
        self.start_time = None
    
    def _create_plotter(self) -> RealTimeStepPlotter:
        """Initialize the real-time plotter."""
        plot_path = self.run_dir.subpath("training_progress.png")
        return RealTimeStepPlotter(
            enabled=self.config.enable_plot,
            plot_interval=self.config.plot_interval,
            out_path=str(plot_path),
            smooth_window=self.config.plot_smooth_window,
            x_axis_mode=self.config.plot_x_axis,
        )
    
    def _create_network_plotter(self) -> NetworkMetricsPlotter:
        """Initialize the network metrics plotter."""
        plot_path = self.run_dir.subpath(NETWORK_PLOT_FILENAME)
        return NetworkMetricsPlotter(
            enabled=NETWORK_PLOT_ENABLED and self.config.enable_plot,
            plot_interval=NETWORK_PLOT_INTERVAL,
            out_path=str(plot_path),
            smooth_window=NETWORK_PLOT_SMOOTH_WINDOW,
            x_axis_mode=self.config.plot_x_axis,
        )
    
    def _create_telecom_plotter(self) -> TelecomNetworkPlotter:
        """Initialize the telecom network topology plotter."""
        plot_path = self.run_dir.subpath(TELECOM_PLOT_FILENAME)
        return TelecomNetworkPlotter(
            scenario=self.scenario,
            agents=self.agents,
            enabled=TELECOM_PLOT_ENABLED and self.config.enable_plot,
            plot_interval=TELECOM_PLOT_INTERVAL,
            out_path=str(plot_path),
            show_coverage=True,
            show_connections=True,
            mobility_manager=self.mobility_manager,
            show_hotspots=MOBILITY_ENABLED,
            show_interference=True,
        )
    
    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop."""
        self.start_time = time.time()
        
        for episode in range(self.config.nepisodes):
            self._run_episode(episode)
            
            # Checkpointing
            if self._should_checkpoint(episode):
                checkpoint_agents(self.run_dir, self.agents, episode + 1)
        
        self._finalize()
        return self.metrics_aggregator.to_dict()
    
    def _run_episode(self, episode: int):
        """Execute a single training episode."""
        ep_start = time.time()
        self.episode_runner.reset()
        
        # Update mobility at episode start (positions update, hotspots age/respawn)
        if self.mobility_manager is not None:
            self.mobility_manager.update_episode()
            # Update agent locations from mobility manager
            for agent in self.agents:
                agent.update_location()
        
        # Set current episode for linear increasing epsilon policy
        self.epsilon_scheduler.set_episode(episode)
        
        while not self.episode_runner.is_done():
            eps = self.epsilon_scheduler.get_epsilon(self.total_steps)
            step_metrics = self.episode_runner.run_step(eps)
            
            train_metrics = self.episode_runner.learn(self.opt)
            
            # Record learning metrics for episode aggregation
            self.episode_runner.metrics.record_learning(train_metrics)
            
            # Step-level logging
            self._log_step(episode, eps, step_metrics, train_metrics)
            
            # Advance state
            self.episode_runner.advance()
            self.total_steps += 1
        
        # Episode complete
        ep_metrics = self.episode_runner.metrics
        self.metrics_aggregator.record_episode(ep_metrics)
        
        # Update telecom network plotter at end of episode
        self._update_telecom_plotter(episode)
        
        ep_duration = time.time() - ep_start
        self._log_episode(episode, ep_metrics, ep_duration)
    
    def _log_step(
        self,
        episode: int,
        epsilon: float,
        step_metrics: Dict,
        train_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log step-level metrics including learning diagnostics."""
        # Get memory size from first agent
        memory_size = len(self.agents[0].memory) if self.agents else 0
        
        # Determine if learning occurred
        did_learn = train_metrics is not None
        
        self.step_logger.log(
            global_step=self.total_steps,
            episode=episode,
            step_in_episode=self.episode_runner.step,
            epsilon=epsilon,
            mean_reward=step_metrics["mean_reward"],
            qos_mean=step_metrics["qos_mean"],
            capacity_sum_mbps=step_metrics["capacity_sum"],
            # Learning diagnostics
            loss=train_metrics.get("loss", "") if train_metrics else "",
            mean_q=train_metrics.get("mean_q", "") if train_metrics else "",
            max_q=train_metrics.get("max_q", "") if train_metrics else "",
            min_q=train_metrics.get("min_q", "") if train_metrics else "",
            q_std=train_metrics.get("q_std", "") if train_metrics else "",
            td_error=train_metrics.get("td_error", "") if train_metrics else "",
            grad_norm=train_metrics.get("grad_norm", "") if train_metrics else "",
            target_q_mean=train_metrics.get("target_q_mean", "") if train_metrics else "",
            memory_size=memory_size,
            did_learn=1 if did_learn else 0,
        )
        
        if self.config.plot_x_axis == "steps":
            self.plotter.update(
                step=self.total_steps,
                epsilon=epsilon,
                mean_reward=step_metrics["mean_reward"],
                qos=step_metrics["qos_mean"],
                capacity_sum_mbps=step_metrics["capacity_sum"],
            )
        
        # Update network metrics plotter if learning occurred
        if train_metrics is not None:
            self.network_plotter.update(
                step=self.total_steps,
                loss=train_metrics.get("loss"),
                mean_q=train_metrics.get("mean_q"),
                max_q=train_metrics.get("max_q"),
                min_q=train_metrics.get("min_q"),
                q_std=train_metrics.get("q_std"),
                grad_norm=train_metrics.get("grad_norm"),
                target_q_mean=train_metrics.get("target_q_mean"),
                td_error=train_metrics.get("td_error"),
            )
    
    def _log_episode(self, episode: int, metrics: EpisodeMetrics, duration: float):
        """Log episode-level metrics including learning diagnostics."""
        eps = self.episode_runner.last_epsilon
        
        # Get memory size from first agent
        memory_size = len(self.agents[0].memory) if self.agents else 0
        metrics.set_memory_size(memory_size)
        
        # CSV logging with all diagnostic metrics
        self.ep_logger.log(
            episode=episode,
            steps=metrics.steps,
            return_val=metrics.return_total,
            avg_reward=metrics.avg_reward,
            reward_min=metrics.reward_min if metrics.reward_min != float('inf') else "",
            reward_max=metrics.reward_max if metrics.reward_max != float('-inf') else "",
            reward_std=metrics.reward_std,
            qos_rate=metrics.qos_rate,
            capacity_mean=metrics.avg_capacity,
            epsilon_last=eps,
            duration_seconds=duration,
            # Learning diagnostics
            loss_avg=metrics.loss_avg if metrics.learning_steps > 0 else "",
            mean_q_avg=metrics.mean_q_avg if metrics.learning_steps > 0 else "",
            max_q_avg=metrics.max_q_avg if metrics.learning_steps > 0 else "",
            td_error_avg=metrics.td_error_avg if metrics.learning_steps > 0 else "",
            grad_norm_avg=metrics.grad_norm_avg if metrics.learning_steps > 0 else "",
            learning_steps=metrics.learning_steps,
            memory_size=memory_size,
            success_rate=metrics.success_rate,
        )
        
        # Console output
        print(
            f"Ep: {episode} | Steps: {metrics.steps} | "
            f"Dur: {duration:.2f}s | Return: {metrics.return_total:.3f} | "
            f"AvgReward: {metrics.avg_reward:.3f} | "
            f"QoSRate: {metrics.qos_rate:.3f} | Eps: {eps:.4f}"
        )
        
        # Episode-mode plotting
        if self.config.plot_x_axis == "episodes":
            self.plotter.update(
                step=episode,
                epsilon=eps,
                mean_reward=metrics.avg_reward,
                qos=metrics.qos_rate,
                capacity_sum_mbps=metrics.avg_capacity,
                episode_return=metrics.return_total,
            )
    
    def _should_checkpoint(self, episode: int) -> bool:
        """Determine if a checkpoint should be saved."""
        interval = self.config.checkpoint_interval
        return interval and (episode + 1) % interval == 0
    
    def _update_telecom_plotter(self, episode: int):
        """Update the telecom network topology plot at end of episode."""
        # Extract current actions as list of integers
        actions = None
        if self.episode_runner.ctx is not None and self.episode_runner.ctx.actions is not None:
            try:
                actions = [int(a.item()) for a in self.episode_runner.ctx.actions]
            except Exception:
                actions = [int(a) for a in self.episode_runner.ctx.actions]
        
        # Compute interference edges for visualization
        interference_edges = []
        if actions:
            # Extract UE positions from agents
            ue_positions = [agent.location.cpu().numpy() for agent in self.agents]
            interference_edges = compute_interference_graph(
                ue_positions=ue_positions,
                actions=actions,
                scenario=self.scenario,
            )
        
        self.telecom_plotter.update(
            step=episode,  # Use episode as the counter since we update per episode
            episode=episode,
            actions=actions,
            interference_edges=interference_edges,
        )
    
    def _finalize(self):
        """Clean up and write final outputs."""
        total_duration = time.time() - self.start_time
        print(f"Total training time: {total_duration/60:.2f} min ({total_duration:.2f} s)")
        
        self.plotter.close()
        self.network_plotter.close()
        self.telecom_plotter.close()
        self.ep_logger.close()
        self.step_logger.close()
        
        # Gather feature flags
        feature_flags = {
            "gnn_enabled": self.gnn_manager.enabled if self.gnn_manager else False,
            "mobility_enabled": self.mobility_manager is not None,
            "gnn_observation_mode": getattr(self.gnn_manager, 'observation_mode', None) if self.gnn_manager else None,
        }
        
        # Write comprehensive experiment summary
        write_experiment_summary(
            run_dir=self.run_dir,
            opt=self.opt,
            sce=self.sce,
            total_episodes=self.config.nepisodes,
            total_steps=self.total_steps,
            total_duration_seconds=total_duration,
            final_metrics=self.metrics_aggregator.get_final_metrics(k=100),
            best_episode=self.metrics_aggregator.get_best_episode(),
            learning_diagnostics=self.metrics_aggregator.get_learning_diagnostics(k=100),
            convergence_info=self.metrics_aggregator.get_convergence_info(),
            feature_flags=feature_flags,
        )
        
        # Final checkpoint
        checkpoint_agents(self.run_dir, self.agents, self.config.nepisodes, tag="final")


class ExperimentManager:
    """Manages experiment setup and execution."""
    
    def __init__(self, opt, sce):
        self.opt = opt
        self.sce = sce
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.scenario = None
        self.run_dir = None
        self.agents = None
        self.config = None
        self.mobility_manager = None
        self.gnn_manager = None
    
    def setup(self):
        """Initialize all components for a trial."""
        self.scenario = Scenario(self.sce)
        self.run_dir = RunDirectoryManager(overwrite=False)
        self.config = TrainingConfig.from_opt(self.opt)
        
        # Save configuration
        save_config(self.run_dir, self.opt, self.sce)
        save_environment_snapshot(self.run_dir)
        
        # Create mobility manager if enabled
        if MOBILITY_ENABLED:
            self.mobility_manager = MobilityManager(
                scenario=self.scenario,
                num_ues=self.opt.nagents,
            )
        else:
            self.mobility_manager = None
        
        # Create GNN manager if enabled
        self.gnn_manager = GNNObservationManager(
            scenario=self.scenario,
            device=self.device,
            enabled=GNN_ENABLED,
        )
        
        # Create agents (with mobility and GNN support)
        gnn_input_dim = self.gnn_manager.output_dim if self.gnn_manager.enabled else None
        self.agents = create_agents(
            self.opt,
            self.sce,
            self.scenario,
            self.device,
            mobility_manager=self.mobility_manager,
            gnn_input_dim=gnn_input_dim,
        )
        
        if self.gnn_manager.enabled:
            print(f"GNN enabled: input_dim={gnn_input_dim}")
    
    def run(self) -> Dict[str, List[float]]:
        """Execute training and return metrics."""
        trainer = Trainer(
            agents=self.agents,
            scenario=self.scenario,
            config=self.config,
            opt=self.opt,
            sce=self.sce,
            run_dir=self.run_dir,
            device=self.device,
            mobility_manager=self.mobility_manager,
            gnn_manager=self.gnn_manager,
        )
        return trainer.train()
    
    def run_trial(self) -> Dict[str, List[float]]:
        """Convenience method: setup and run."""
        self.setup()
        return self.run()


# Convenience function for backward compatibility
def run_trial(opt, sce):
    """Run a single trial (backward compatible with original API)."""
    experiment = ExperimentManager(opt, sce)
    return experiment.run_trial()