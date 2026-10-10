"""Double-DQN optimization, target updates, and learning diagnostics."""
from dataclasses import dataclass
from typing import Dict, Union
import torch
from torch.nn import functional as F
from functions.configuration import config_value
from constants import GRAD_CLIP_VALUE
from .memory import ReplayMemory, TensorReplayMemory, Transition

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


class DQNOptimizer:
    """Handles DQN optimization logic."""
    
    def __init__(self, model_policy, model_target, optimizer, opt, device):
        self.model_policy = model_policy
        self.model_target = model_target
        self.optimizer = optimizer
        self.opt = opt
        self.device = device
        from functions.gpu_manager import GPUManager
        self._amp_enabled, self._scaler = GPUManager.build_amp_context(opt, device)
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

        if len(memory) < config_value(self.opt, "min_memory_for_learning", 0):
            return self._last_metrics

        sample_out = memory.Sample(self.opt.batch_size)

        if isinstance(sample_out, tuple) and len(sample_out) == 5 and torch.is_tensor(sample_out[0]):
            # TensorReplayMemory path: already batched tensors.
            state_batch, action_batch, next_state_batch, reward_batch, terminated_batch = sample_out
            terminated_batch = terminated_batch.to(self.device)
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
            terminated_batch = torch.tensor(batch.terminated, device=self.device, dtype=torch.bool)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self._amp_enabled):
            q_all = self.model_policy(state_batch)
            q_pred = q_all.gather(1, action_batch)

            with torch.no_grad():
                next_actions = self.model_policy(next_state_batch).argmax(1, keepdim=True)
                q_next = self.model_target(next_state_batch).gather(1, next_actions)
                if not config_value(self.opt, "bootstrap_on_termination", True):
                    q_next = q_next.masked_fill(terminated_batch.unsqueeze(1), 0.)
                target_q = reward_batch.unsqueeze(1) + self.opt.gamma * q_next

            td_error = (q_pred - target_q).abs()
            loss = F.smooth_l1_loss(q_pred, target_q)

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DQN loss")
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
                    (q_all.std() if q_all.numel() > 1 else q_all.new_zeros(())).detach(),
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

