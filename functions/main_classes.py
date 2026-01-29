import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

import torch

from telecom.scenario import Scenario
from rl.training import (
    select_actions,
    compute_rewards_and_next_state,
    store_and_learn,
    create_agents,
    initialize_episode,
    should_terminate,
)
from functions.live_plot import RealTimeStepPlotter
from functions.logging import (
    RunDirectoryManager,
    save_config,
    save_environment_snapshot,
    EpisodeMetricsLogger,
    StepMetricsLogger,
    checkpoint_agents,
    write_summary,
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
                 eps_decay_steps: int, nepisodes: int, nsteps: int):
        self.policy = policy
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_increment = eps_increment
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.nepisodes = nepisodes
        self.nsteps = nsteps
        self.current_episode = 0
        self.current_step_in_episode = 0
    
    def set_episode(self, episode: int):
        """Set current episode (for linear increasing policy)."""
        self.current_episode = episode
        self.current_step_in_episode = 0
    
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
        return cls(
            policy=getattr(opt, "epsilon_policy", "exponential_decay"),
            eps_min=getattr(opt, "eps_min", 0.0),
            eps_max=getattr(opt, "eps_max", 0.9),
            eps_increment=getattr(opt, "eps_increment", 0.003),
            eps_start=getattr(opt, "eps_start", 1.0),
            eps_end=getattr(opt, "eps_end", 0.05),
            eps_decay_steps=getattr(opt, "eps_decay_steps", 10000),
            nepisodes=getattr(opt, "nepisodes", 1000),
            nsteps=getattr(opt, "nsteps", 500),
        )


@dataclass
class EpisodeMetrics:
    """Accumulates metrics within a single episode."""
    reward_sum: float = 0.0
    return_total: float = 0.0
    qos_sum: float = 0.0
    capacity_sum: float = 0.0
    steps: int = 0
    
    def record_step(self, rewards: torch.Tensor, qos: torch.Tensor, capacity: torch.Tensor):
        """Record metrics from a single step."""
        self.reward_sum += float(rewards.mean().item())
        self.return_total += float(rewards.sum().item())
        self.qos_sum += float(qos.mean().item())
        self.capacity_sum += float(capacity.sum().item())
        self.steps += 1
    
    @property
    def avg_reward(self) -> float:
        return self.reward_sum / max(1, self.steps)
    
    @property
    def qos_rate(self) -> float:
        return self.qos_sum / max(1, self.steps)
    
    @property
    def avg_capacity(self) -> float:
        return self.capacity_sum / max(1, self.steps)


class MetricsAggregator:
    """Aggregates metrics across all episodes."""
    
    def __init__(self):
        self.avg_rewards: List[float] = []
        self.returns: List[float] = []
        self.qos_rates: List[float] = []
    
    def record_episode(self, metrics: EpisodeMetrics):
        """Record metrics from a completed episode."""
        self.avg_rewards.append(metrics.avg_reward)
        self.returns.append(metrics.return_total)
        self.qos_rates.append(metrics.qos_rate)
    
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
    ):
        self.agents = agents
        self.scenario = scenario
        self.config = config
        self.device = device
        self.action_tail_len = action_tail_len
        self.state_target = torch.ones(config.nagents, device=device)
        
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
    
    def run_step(self, epsilon: float) -> Dict[str, float]:
        """Execute a single step and return step metrics."""
        self.last_epsilon = epsilon
        
        # Select actions
        self.ctx.actions = select_actions(
            self.agents, self.ctx.state, self.scenario, epsilon
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
    
    def learn(self, opt):
        """Store experience and update networks."""
        store_and_learn(
            self.agents,
            self.ctx.state,
            self.ctx.actions,
            self.ctx.next_state,
            self.ctx.rewards,
            self.scenario,
            self.step,
            opt,
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
        run_dir: RunDirectoryManager,
        device: torch.device,
    ):
        self.agents = agents
        self.scenario = scenario
        self.config = config
        self.opt = opt
        self.run_dir = run_dir
        self.device = device
        
        # Components
        self.epsilon_scheduler = EpsilonScheduler.from_opt(opt)
        self.episode_runner = EpisodeRunner(agents, scenario, config, device)
        self.metrics_aggregator = MetricsAggregator()
        
        # Loggers and plotter
        self.plotter = self._create_plotter()
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
        
        # Set current episode for linear increasing epsilon policy
        self.epsilon_scheduler.set_episode(episode)
        
        while not self.episode_runner.is_done():
            eps = self.epsilon_scheduler.get_epsilon(self.total_steps)
            step_metrics = self.episode_runner.run_step(eps)
            
            # Learn from experience
            self.episode_runner.learn(self.opt)
            
            # Step-level logging
            self._log_step(episode, eps, step_metrics)
            
            # Advance state
            self.episode_runner.advance()
            self.total_steps += 1
        
        # Episode complete
        ep_metrics = self.episode_runner.metrics
        self.metrics_aggregator.record_episode(ep_metrics)
        
        ep_duration = time.time() - ep_start
        self._log_episode(episode, ep_metrics, ep_duration)
    
    def _log_step(self, episode: int, epsilon: float, step_metrics: Dict):
        """Log step-level metrics."""
        self.step_logger.log(
            global_step=self.total_steps,
            episode=episode,
            step_in_episode=self.episode_runner.step,
            epsilon=epsilon,
            mean_reward=step_metrics["mean_reward"],
            qos_mean=step_metrics["qos_mean"],
            capacity_sum_mbps=step_metrics["capacity_sum"],
        )
        
        if self.config.plot_x_axis == "steps":
            self.plotter.update(
                step=self.total_steps,
                epsilon=epsilon,
                mean_reward=step_metrics["mean_reward"],
                qos=step_metrics["qos_mean"],
                capacity_sum_mbps=step_metrics["capacity_sum"],
            )
    
    def _log_episode(self, episode: int, metrics: EpisodeMetrics, duration: float):
        """Log episode-level metrics."""
        eps = self.episode_runner.last_epsilon
        
        # CSV logging
        self.ep_logger.log(
            episode=episode,
            steps=metrics.steps,
            avg_reward=metrics.avg_reward,
            return_val=metrics.return_total,
            qos_rate=metrics.qos_rate,
            capacity_mean=metrics.avg_capacity,
            epsilon_last=eps,
            duration_seconds=duration,
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
    
    def _finalize(self):
        """Clean up and write final outputs."""
        total_duration = time.time() - self.start_time
        print(f"Total training time: {total_duration/60:.2f} min ({total_duration:.2f} s)")
        
        # Close loggers
        self.plotter.close()
        self.ep_logger.close()
        self.step_logger.close()
        
        # Write summary
        summary = {
            "episodes": self.config.nepisodes,
            "total_steps": self.total_steps,
            "total_training_seconds": total_duration,
            **self.metrics_aggregator.get_summary(k=100),
        }
        write_summary(self.run_dir, summary)
        
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
    
    def setup(self):
        """Initialize all components for a trial."""
        self.scenario = Scenario(self.sce)
        self.run_dir = RunDirectoryManager(overwrite=False)
        self.config = TrainingConfig.from_opt(self.opt)
        
        # Save configuration
        save_config(self.run_dir, self.opt, self.sce)
        save_environment_snapshot(self.run_dir)
        
        # Create agents
        self.agents = create_agents(self.opt, self.sce, self.scenario, self.device)
    
    def run(self) -> Dict[str, List[float]]:
        """Execute training and return metrics."""
        trainer = Trainer(
            agents=self.agents,
            scenario=self.scenario,
            config=self.config,
            opt=self.opt,
            run_dir=self.run_dir,
            device=self.device,
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