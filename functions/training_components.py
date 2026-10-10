"""Training settings, exploration schedules, and shared metric definitions."""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np
import torch
from constants import DEFAULT_PLOT_INTERVAL, DEFAULT_STEP_LOG_THROTTLE, EPS_MAX_DECAY, EPS_MAX_MIN

@dataclass
class TrainingConfig:
    """Typed configuration with defaults for training parameters."""
    nepisodes: int
    nsteps: int
    nagents: int
    enable_plot: bool = True
    plot_interval: int = DEFAULT_PLOT_INTERVAL
    plot_smooth_window: int = 50
    plot_x_axis: str = "episodes"
    early_termination: bool = True
    step_log_throttle: int = DEFAULT_STEP_LOG_THROTTLE  # Higher = faster, less CSV I/O
    checkpoint_interval: int = 0  # Completed episodes between policy snapshots
    # Epsilon schedule
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    learn_every: int = 1  # Run optimizer every N environment steps (1 = every step)

    @classmethod
    def from_opt(cls, opt) -> "TrainingConfig":
        """Create config from DotDic/namespace object with defaults."""
        return cls(
            nepisodes=opt.nepisodes,
            nsteps=opt.nsteps,
            nagents=opt.nagents,
            enable_plot=getattr(opt, "enable_plot", True),
            plot_interval=getattr(opt, "plot_interval", DEFAULT_PLOT_INTERVAL),
            plot_smooth_window=getattr(opt, "plot_smooth_window", 50),
            plot_x_axis=getattr(opt, "plot_x_axis", "episodes"),
            early_termination=getattr(opt, "early_termination", True),
            step_log_throttle=getattr(opt, "step_log_throttle", DEFAULT_STEP_LOG_THROTTLE),
            checkpoint_interval=getattr(opt, "checkpoint_interval", 0),
            learn_every=getattr(opt, "learn_every", 1),
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
            eps = self.eps_min + self.eps_increment * self.current_step_in_episode
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
    num_agents: int = 0

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

    # Cell-free specific episode-level aggregates
    mean_soft_qos_sum: float = 0.0
    min_soft_qos_sum: float = 0.0
    jain_fairness_sum: float = 0.0
    min_rate_sum: float = 0.0
    max_rate_sum: float = 0.0
    p5_rate_sum: float = 0.0
    p95_rate_sum: float = 0.0
    sinr_db_mean_sum: float = 0.0
    interference_mean_sum: float = 0.0
    ap_load_mean_sum: float = 0.0

    def record_step(
        self,
        rewards: torch.Tensor,
        qos: torch.Tensor,
        capacity: torch.Tensor,
        cellfree_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Record environment metrics from a single step."""
        if self.num_agents <= 0:
            self.num_agents = int(rewards.numel())

        stat_vec = torch.stack(
            [rewards.mean(), rewards.sum(), qos.mean(), capacity.sum()]
        ).detach().cpu()
        mean_reward = float(stat_vec[0])
        return_total = float(stat_vec[1])
        qos_mean = float(stat_vec[2])
        capacity_sum = float(stat_vec[3])

        self.reward_sum += mean_reward
        self.return_total += return_total
        self.qos_sum += qos_mean
        self.capacity_sum += capacity_sum

        # Track reward distribution
        self.reward_min = min(self.reward_min, mean_reward)
        self.reward_max = max(self.reward_max, mean_reward)
        self.reward_squared_sum += mean_reward ** 2

        # Track QoS success (full satisfaction)
        if qos_mean >= 1.0:
            self.qos_success_count += 1

        if cellfree_metrics:
            self.mean_soft_qos_sum += float(cellfree_metrics.get("mean_soft_qos", 0.0) or 0.0)
            self.min_soft_qos_sum += float(cellfree_metrics.get("min_soft_qos", 0.0) or 0.0)
            self.jain_fairness_sum += float(cellfree_metrics.get("jain_fairness", 0.0) or 0.0)
            self.min_rate_sum += float(cellfree_metrics.get("min_rate_mbps", 0.0) or 0.0)
            self.max_rate_sum += float(cellfree_metrics.get("max_rate_mbps", 0.0) or 0.0)
            self.p5_rate_sum += float(cellfree_metrics.get("rate_p5_mbps", 0.0) or 0.0)
            self.p95_rate_sum += float(cellfree_metrics.get("rate_p95_mbps", 0.0) or 0.0)
            self.sinr_db_mean_sum += float(np.mean(cellfree_metrics.get("sinr_db", []) or [0.0]))
            self.interference_mean_sum += float(cellfree_metrics.get("interference_mean_mw", 0.0) or 0.0)
            self.ap_load_mean_sum += float(cellfree_metrics.get("ap_load_mean", 0.0) or 0.0)

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
    def return_per_step(self) -> float:
        return self.return_total / max(1, self.steps)

    @property
    def return_per_agent_step(self) -> float:
        return self.return_total / max(1, self.steps * max(1, self.num_agents))

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

    @property
    def mean_soft_qos(self) -> float:
        return self.mean_soft_qos_sum / max(1, self.steps)

    @property
    def min_soft_qos(self) -> float:
        return self.min_soft_qos_sum / max(1, self.steps)

    @property
    def jain_fairness(self) -> float:
        return self.jain_fairness_sum / max(1, self.steps)

    @property
    def min_rate_mbps(self) -> float:
        return self.min_rate_sum / max(1, self.steps)

    @property
    def max_rate_mbps(self) -> float:
        return self.max_rate_sum / max(1, self.steps)

    @property
    def rate_p5_mbps(self) -> float:
        return self.p5_rate_sum / max(1, self.steps)

    @property
    def rate_p95_mbps(self) -> float:
        return self.p95_rate_sum / max(1, self.steps)

    @property
    def sinr_db_mean(self) -> float:
        return self.sinr_db_mean_sum / max(1, self.steps)

    @property
    def interference_mean_mw(self) -> float:
        return self.interference_mean_sum / max(1, self.steps)

    @property
    def ap_load_mean(self) -> float:
        return self.ap_load_mean_sum / max(1, self.steps)


class MetricsAggregator:
    """Aggregates metrics across all episodes."""

    def __init__(self, convergence_window: int = 50, convergence_threshold: float = 0.01):
        self.avg_rewards: List[float] = []
        self.returns: List[float] = []
        self.returns_per_agent_step: List[float] = []
        self.qos_rates: List[float] = []
        self.soft_qos_means: List[float] = []
        self.jain_fairness: List[float] = []
        self.min_rates_mbps: List[float] = []
        self.p5_rates_mbps: List[float] = []
        self.p95_rates_mbps: List[float] = []

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
        self.returns_per_agent_step.append(metrics.return_per_agent_step)
        self.qos_rates.append(metrics.qos_rate)
        self.soft_qos_means.append(metrics.mean_soft_qos)
        self.jain_fairness.append(metrics.jain_fairness)
        self.min_rates_mbps.append(metrics.min_rate_mbps)
        self.p5_rates_mbps.append(metrics.rate_p5_mbps)
        self.p95_rates_mbps.append(metrics.rate_p95_mbps)

        # Track best episode (by return)
        episode_idx = len(self.returns) - 1
        if metrics.return_total > self.best_return:
            self.best_return = metrics.return_total
            self.best_episode = {
                "episode": episode_idx,
                "return": metrics.return_total,
                "return_per_agent_step": metrics.return_per_agent_step,
                "avg_reward": metrics.avg_reward,
                "qos_rate": metrics.qos_rate,
                "steps": metrics.steps,
                "soft_qos": metrics.mean_soft_qos,
                "jain_fairness": metrics.jain_fairness,
                "min_rate_mbps": metrics.min_rate_mbps,
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
            "mean_last_100_return_per_agent_step": self.mean_last_k(self.returns_per_agent_step, k),
            "mean_last_100_qos": self.mean_last_k(self.qos_rates, k),
            "mean_last_100_soft_qos": self.mean_last_k(self.soft_qos_means, k),
            "mean_last_100_jain_fairness": self.mean_last_k(self.jain_fairness, k),
            "mean_last_100_min_rate_mbps": self.mean_last_k(self.min_rates_mbps, k),
            "mean_last_100_rate_p5_mbps": self.mean_last_k(self.p5_rates_mbps, k),
            "mean_last_100_rate_p95_mbps": self.mean_last_k(self.p95_rates_mbps, k),
        }

    def get_final_metrics(self, k: int = 100) -> Dict[str, Any]:
        """Get comprehensive final metrics for experiment summary."""
        return {
            "mean_reward": self.mean_last_k(self.avg_rewards, k),
            "mean_return": self.mean_last_k(self.returns, k),
            "mean_return_per_agent_step": self.mean_last_k(self.returns_per_agent_step, k),
            "mean_qos": self.mean_last_k(self.qos_rates, k),
            "mean_soft_qos": self.mean_last_k(self.soft_qos_means, k),
            "mean_jain_fairness": self.mean_last_k(self.jain_fairness, k),
            "mean_min_rate_mbps": self.mean_last_k(self.min_rates_mbps, k),
            "mean_rate_p5_mbps": self.mean_last_k(self.p5_rates_mbps, k),
            "mean_rate_p95_mbps": self.mean_last_k(self.p95_rates_mbps, k),
            "std_reward": self._std_last_k(self.avg_rewards, k),
            "std_return": self._std_last_k(self.returns, k),
            "std_return_per_agent_step": self._std_last_k(self.returns_per_agent_step, k),
            "std_qos": self._std_last_k(self.qos_rates, k),
            "std_soft_qos": self._std_last_k(self.soft_qos_means, k),
            "std_jain_fairness": self._std_last_k(self.jain_fairness, k),
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
            "return_per_agent_step": self.returns_per_agent_step,
        }
