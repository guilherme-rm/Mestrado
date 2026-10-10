from functions.training_components import TrainingConfig, EpsilonScheduler, EpisodeMetrics, MetricsAggregator
import time
from collections import deque
from typing import List, Dict, Optional, Any

import numpy as np
import torch
from functions.gpu_manager import GPUManager
from functions.configuration import resolve_configuration
from functions.reproducibility import resolve_seeds

from telecom.environment_factory import create_scenario, resolve_environment_type
from telecom.mobility import MobilityManager
from telecom.interference import compute_interference_graph
from rl.interaction import EnvironmentInteraction, movement_due
from rl.metrics import qos_summary
from rl.training import (
    get_flat_state_dim,
    select_actions,
    store_and_learn,
    create_agents,
    initialize_episode,
    should_terminate,
    GNNObservationManager,
)
from functions.live_plot import RealTimeStepPlotter
from functions.network_metrics_plot import NetworkMetricsPlotter
from functions.telecom_network_plot import TelecomNetworkPlotter
from functions.resource_metrics_plot import ResourceMetricsPlotter
from functions.logging import (
    RunDirectoryManager,
    save_config,
    save_environment_snapshot,
    EpisodeMetricsLogger,
    StepMetricsLogger,
    TimeMetricsLogger,
    write_summary,
    write_experiment_summary,
)
from constants import (
    NETWORK_PLOT_INTERVAL,
    NETWORK_PLOT_SMOOTH_WINDOW,
    NETWORK_PLOT_FILENAME,
    TELECOM_PLOT_INTERVAL,
    TELECOM_PLOT_FILENAME,
    RESOURCE_PLOT_INTERVAL,
    RESOURCE_PLOT_SMOOTH_WINDOW,
    RESOURCE_PLOT_FILENAME,
)


class EpisodeRunner:
    """Handles execution of a single episode."""
    
    def __init__(
        self,
        agents: List,
        scenario: Any,
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
        self.interaction = EnvironmentInteraction(agents, scenario, gnn_manager)
        
        # Episode state (reset each episode)
        self.ctx = None
        self.metrics = None
        self.last_actions = None
        self.step = 0
        self.last_epsilon = 0.0
        self._terminated_early = False
    
    def reset(self):
        """Initialize state for a new episode."""
        state_dim = self.config.nagents * 6 if hasattr(self.scenario, "begin_step") else self.config.nagents
        self.ctx = initialize_episode(self.config.nagents, self.device, state_dim=state_dim)
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
        self._last_capacity = None
    
    def run_step(self, epsilon: float) -> Dict[str, Any]:
        """Execute a single step and return step metrics."""
        step_t0 = time.perf_counter()
        self.last_epsilon = epsilon
        
        # Get GNN observations if enabled
        gnn_obs = None
        t0 = time.perf_counter()
        gnn_obs = self.interaction.observe(
            self.ctx.state, self.ctx.actions if self.step > 0 else None, self._last_capacity)
        # The existing per-agent observation/replay path also handles enriched vectors.
        self._gnn_obs = gnn_obs
        gnn_encode_seconds = time.perf_counter() - t0
        
        # Select actions (with GNN observations if available)
        t0 = time.perf_counter()
        self.ctx.actions, select_actions_timing = select_actions(
            self.agents, self.ctx.state, self.scenario, epsilon,
            gnn_observations=gnn_obs,
            return_timing=True,
        )
        select_actions_seconds = time.perf_counter() - t0
        
        # Record actions for debugging
        self._record_actions()
        
        # Environment step
        t0 = time.perf_counter()
        self.ctx.rewards, self.ctx.qos, self.ctx.next_state, capacity, cellfree_step_metrics = (
            self.interaction.apply_joint_action(self.ctx.actions, self.ctx.state)
        )
        self._last_capacity = capacity
        telecom_update_seconds = time.perf_counter() - t0
        
        # Record metrics
        t0 = time.perf_counter()
        self.metrics.record_step(self.ctx.rewards, self.ctx.qos, capacity, cellfree_step_metrics)
        record_step_metrics_seconds = time.perf_counter() - t0
        
        # Prepare step metrics for logging
        step_stat_vec = torch.stack(
            [self.ctx.rewards.mean(), capacity.sum()]
        ).detach().cpu()

        step_metrics = {
            "mean_reward": float(step_stat_vec[0]),
            "qos_mean": qos_summary(self.ctx.qos)['qos_fraction'],
            "capacity_sum": float(step_stat_vec[1]),
            "cellfree": cellfree_step_metrics,
            "timing": {
                "run_step_seconds": time.perf_counter() - step_t0,
                "gnn_encode_seconds": gnn_encode_seconds,
                "select_actions_seconds": select_actions_seconds,
                "select_actions_forward_seconds": select_actions_timing.get("select_actions_forward_seconds", 0.0),
                "select_actions_policy_seconds": select_actions_timing.get("select_actions_policy_seconds", 0.0),
                "telecom_update_seconds": telecom_update_seconds,
                "record_step_metrics_seconds": record_step_metrics_seconds,
            },
        }
        
        return step_metrics
    
    def learn(self, opt, global_step=0) -> tuple[Optional[Dict[str, float]], Dict[str, float]]:
        """Store experience and update networks.
        
        Returns:
            Aggregated training metrics if learning occurred, None otherwise.
        """
        # Get next GNN observations for storing transitions
        gnn_obs_next = None
        gnn_obs_next = self.interaction.observe(
            self.ctx.next_state, self.ctx.actions, self._last_capacity)

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
            terminated=self.config.early_termination and should_terminate(self.ctx.next_state, self.state_target),
            optimize=(global_step % self.config.learn_every == 0),
            target_step=global_step if getattr(opt, "target_update_clock", "episode") == "global" else None,
        )

    def move_users(self, mover):
        self.interaction.move_users(mover)
    
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
            if isinstance(self.ctx.actions, torch.Tensor):
                action_values = self.ctx.actions.detach().cpu().tolist()
            else:
                action_values = list(self.ctx.actions)
            for i in range(self.config.nagents):
                try:
                    self.last_actions[i].append(int(action_values[i]))
                except Exception:
                    self.last_actions[i].append(0)


class Trainer:
    """Orchestrates the training loop across episodes."""
    
    def __init__(
        self,
        agents: List,
        scenario: Any,
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
        self.environment_type = resolve_environment_type(opt)
        
        # Components
        self.epsilon_scheduler = EpsilonScheduler.from_opt(opt)
        self.episode_runner = EpisodeRunner(agents, scenario, config, device, gnn_manager=gnn_manager)
        self.metrics_aggregator = MetricsAggregator()
        
        self.plotter = self._create_plotter()
        self.network_plotter = self._create_network_plotter()
        self.telecom_plotter = self._create_telecom_plotter()
        self.resource_plotter = self._create_resource_plotter()
        self.ep_logger = EpisodeMetricsLogger(run_dir)
        self.step_logger = StepMetricsLogger(run_dir, throttle=config.step_log_throttle)
        self.time_logger = TimeMetricsLogger(run_dir)
        
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
            deferred=self.opt.deferred_plotting,
        )
    
    def _create_network_plotter(self) -> NetworkMetricsPlotter:
        """Initialize the network metrics plotter."""
        plot_path = self.run_dir.subpath(NETWORK_PLOT_FILENAME)
        return NetworkMetricsPlotter(
            enabled=self.opt.network_plot_enabled and self.config.enable_plot,
            plot_interval=NETWORK_PLOT_INTERVAL,
            out_path=str(plot_path),
            smooth_window=NETWORK_PLOT_SMOOTH_WINDOW,
            x_axis_mode=self.config.plot_x_axis,
            deferred=self.opt.deferred_plotting,
        )
    
    def _create_telecom_plotter(self) -> TelecomNetworkPlotter:
        """Initialize the telecom network topology plotter."""
        plot_path = self.run_dir.subpath(TELECOM_PLOT_FILENAME)
        return TelecomNetworkPlotter(
            scenario=self.scenario,
            agents=self.agents,
            enabled=self.opt.telecom_plot_enabled and self.config.enable_plot,
            plot_interval=TELECOM_PLOT_INTERVAL,
            out_path=str(plot_path),
            show_coverage=True,
            show_connections=True,
            mobility_manager=self.mobility_manager,
            show_hotspots=self.mobility_manager is not None,
            show_interference=True,
            environment_type=self.environment_type,
            deferred=self.opt.deferred_plotting,
        )
    
    def _create_resource_plotter(self) -> ResourceMetricsPlotter:
        """Initialize the resource metrics plotter."""
        plot_path = self.run_dir.subpath(RESOURCE_PLOT_FILENAME)
        return ResourceMetricsPlotter(
            enabled=self.opt.resource_plot_enabled and self.config.enable_plot,
            plot_interval=RESOURCE_PLOT_INTERVAL,
            out_path=str(plot_path),
            smooth_window=RESOURCE_PLOT_SMOOTH_WINDOW,
            x_axis_mode=self.config.plot_x_axis,
            deferred=self.opt.deferred_plotting,
        )
    
    def train(self) -> Dict[str, List[float]]:
        import json
        status = {"state": "running", "plot_errors": []}
        status_path = self.run_dir.subpath("run_status.json")
        status_path.write_text(json.dumps(status, indent=2))
        self.plot_errors = status["plot_errors"]
        try:
            result = self._train()
            status["state"] = "completed"
            return result
        except BaseException as exc:
            status.update(state="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                          error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            cleanup_errors = []
            for logger in (self.ep_logger, self.step_logger, self.time_logger):
                try:
                    logger.close()
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            if cleanup_errors:
                status["cleanup_errors"] = cleanup_errors
                status["state"] = "failed"
            status["total_steps"] = self.total_steps
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2))
            temporary.replace(status_path)
            if cleanup_errors:
                raise OSError(f"Could not flush run logs: {cleanup_errors}")

    def _train(self) -> Dict[str, List[float]]:
        """Run the full training loop."""
        self.start_time = time.time()
        interval = int(getattr(self.opt, "checkpoint_interval", None) or 0)
        save_final = getattr(self.opt, "save_final_checkpoint", None)
        if save_final is None:
            save_final = self.environment_type == "hetnet"
        if interval < 0:
            raise ValueError("checkpoint_interval must be nonnegative")
        if self.environment_type != "hetnet":
            if save_final:
                raise ValueError("Policy checkpoints currently support HetNet only")
            if interval:
                print("Policy checkpoints are HetNet-only; ignoring checkpoint_interval for this environment.")
                interval = 0
        if self.config.nepisodes < 1:
            raise ValueError("nepisodes must be positive")
        for episode in range(self.config.nepisodes):
            self._run_episode(episode)
            if interval and (episode + 1) % interval == 0:
                self._save_policy_checkpoint(f"episode_{episode + 1:05d}.pt", episode)
        if save_final:
            self._save_policy_checkpoint("final.pt", self.config.nepisodes - 1)
        self._finalize()
        return self.metrics_aggregator.to_dict()
    
    def _save_policy_checkpoint(self, filename, episode):
        from rl.checkpoints import save_checkpoint
        path = self.run_dir.subpath("checkpoints") / filename
        save_checkpoint(path, self.agents, self.scenario, self.opt, self.sce,
                        self.gnn_manager, episode, self.total_steps)
        print(f"Saved policy checkpoint: {path}", flush=True)

    def _run_episode(self, episode: int):
        """Execute a single training episode."""
        ep_start = time.time()
        self.episode_runner.reset()
        
        # Update mobility at episode start (positions update, hotspots age/respawn)
        if self.mobility_manager is not None:
            self.episode_runner.move_users(self.mobility_manager)
        
        # Set current episode for linear increasing epsilon policy
        self.epsilon_scheduler.set_episode(episode)
        if hasattr(self.scenario, "set_curriculum_progress"):
            self.scenario.set_curriculum_progress(episode, self.config.nepisodes)

        telecom_plotter_update_seconds = 0.0
        
        move_every = int(getattr(self.opt, "mobility_decisions_per_move", 0) or 0)

        while not self.episode_runner.is_done():
            step_t0 = time.perf_counter()

            t0 = time.perf_counter()
            eps = self.epsilon_scheduler.get_epsilon(self.total_steps)
            epsilon_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            step_metrics = self.episode_runner.run_step(eps)
            run_step_seconds = time.perf_counter() - t0

            timing = step_metrics.get("timing", {})

            # Move after the reward so the stored next observation uses the geometry the next action sees.
            next_step = self.episode_runner.step + 1
            if (self.mobility_manager is not None
                    and movement_due(next_step, move_every, self.config.nsteps)):
                self.episode_runner.move_users(self.mobility_manager)

            t0 = time.perf_counter()
            train_metrics, learn_timing = self.episode_runner.learn(self.opt, global_step=self.total_steps)
            learn_seconds = time.perf_counter() - t0
            
            # Record learning metrics for episode aggregation
            t0 = time.perf_counter()
            self.episode_runner.metrics.record_learning(train_metrics)
            if train_metrics is not None:
                self.metrics_aggregator.record_learning_step(
                    loss=train_metrics.get("loss"),
                    mean_q=train_metrics.get("mean_q"),
                    td_error=train_metrics.get("td_error"),
                    grad_norm=train_metrics.get("grad_norm"),
                )
            record_learning_seconds = time.perf_counter() - t0
            
            # Step-level logging
            t0 = time.perf_counter()
            self._log_step(episode, eps, step_metrics, train_metrics)
            log_step_seconds = time.perf_counter() - t0

            step_in_episode = self.episode_runner.step
            did_learn = 1 if train_metrics is not None else 0
            
            # Advance state
            t0 = time.perf_counter()
            self.episode_runner.advance()
            advance_seconds = time.perf_counter() - t0

            self.time_logger.log(
                global_step=self.total_steps,
                episode=episode,
                step_in_episode=step_in_episode,
                did_learn=did_learn,
                epsilon_seconds=epsilon_seconds,
                run_step_seconds=run_step_seconds,
                gnn_encode_seconds=timing.get("gnn_encode_seconds", ""),
                select_actions_seconds=timing.get("select_actions_seconds", ""),
                select_actions_forward_seconds=timing.get("select_actions_forward_seconds", ""),
                select_actions_policy_seconds=timing.get("select_actions_policy_seconds", ""),
                telecom_update_seconds=timing.get("telecom_update_seconds", ""),
                record_step_metrics_seconds=timing.get("record_step_metrics_seconds", ""),
                learn_seconds=learn_seconds,
                store_and_learn_total_seconds=learn_timing.get("store_and_learn_total_seconds", ""),
                store_transition_seconds=learn_timing.get("store_transition_seconds", ""),
                detach_seconds=learn_timing.get("detach_seconds", ""),
                optimize_seconds=learn_timing.get("optimize_seconds", ""),
                target_update_seconds=learn_timing.get("target_update_seconds", ""),
                sample_batch_seconds=train_metrics.get("sample_batch_seconds", "") if train_metrics else "",
                batch_to_device_seconds=train_metrics.get("batch_to_device_seconds", "") if train_metrics else "",
                q_pred_seconds=train_metrics.get("q_pred_seconds", "") if train_metrics else "",
                target_q_seconds=train_metrics.get("target_q_seconds", "") if train_metrics else "",
                loss_backward_seconds=train_metrics.get("loss_backward_seconds", "") if train_metrics else "",
                grad_norm_seconds=train_metrics.get("grad_norm_seconds", "") if train_metrics else "",
                optimizer_step_seconds=train_metrics.get("optimizer_step_seconds", "") if train_metrics else "",
                optimize_total_seconds=train_metrics.get("optimize_total_seconds", "") if train_metrics else "",
                record_learning_seconds=record_learning_seconds,
                log_step_seconds=log_step_seconds,
                advance_seconds=advance_seconds,
                step_total_seconds=time.perf_counter() - step_t0,
                telecom_plotter_update_seconds="",
            )
            self.total_steps += 1
        
        # Episode complete
        ep_metrics = self.episode_runner.metrics
        self.metrics_aggregator.record_episode(ep_metrics)
        
        # Update telecom network plotter at end of episode
        t0 = time.perf_counter()
        self._update_telecom_plotter(episode)
        telecom_plotter_update_seconds = time.perf_counter() - t0

        self.time_logger.log(
            global_step=self.total_steps,
            episode=episode,
            step_in_episode="episode_end",
            did_learn="",
            epsilon_seconds="",
            run_step_seconds="",
            gnn_encode_seconds="",
            select_actions_seconds="",
            select_actions_forward_seconds="",
            select_actions_policy_seconds="",
            telecom_update_seconds="",
            record_step_metrics_seconds="",
            learn_seconds="",
            store_and_learn_total_seconds="",
            store_transition_seconds="",
            detach_seconds="",
            optimize_seconds="",
            target_update_seconds="",
            sample_batch_seconds="",
            batch_to_device_seconds="",
            q_pred_seconds="",
            target_q_seconds="",
            loss_backward_seconds="",
            grad_norm_seconds="",
            optimizer_step_seconds="",
            optimize_total_seconds="",
            record_learning_seconds="",
            log_step_seconds="",
            advance_seconds="",
            step_total_seconds="",
            telecom_plotter_update_seconds=telecom_plotter_update_seconds,
        )
        
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
            mean_soft_qos=step_metrics.get("cellfree", {}).get("mean_soft_qos", ""),
            min_soft_qos=step_metrics.get("cellfree", {}).get("min_soft_qos", ""),
            jain_fairness=step_metrics.get("cellfree", {}).get("jain_fairness", ""),
            min_rate_mbps=step_metrics.get("cellfree", {}).get("min_rate_mbps", ""),
            max_rate_mbps=step_metrics.get("cellfree", {}).get("max_rate_mbps", ""),
            rate_p5_mbps=step_metrics.get("cellfree", {}).get("rate_p5_mbps", ""),
            rate_p95_mbps=step_metrics.get("cellfree", {}).get("rate_p95_mbps", ""),
            sinr_db_mean=float(np.mean(step_metrics.get("cellfree", {}).get("sinr_db", []) or [0.0])) if step_metrics.get("cellfree") else "",
            interference_mean_mw=step_metrics.get("cellfree", {}).get("interference_mean_mw", ""),
            ap_load_mean=step_metrics.get("cellfree", {}).get("ap_load_mean", ""),
        )
        
        if self.config.plot_x_axis == "steps":
            cf = step_metrics.get("cellfree", {})
            self.plotter.update(
                step=self.total_steps,
                epsilon=epsilon,
                mean_reward=step_metrics["mean_reward"],
                qos=step_metrics["qos_mean"],
                capacity_sum_mbps=step_metrics["capacity_sum"],
                mean_soft_qos=cf.get("mean_soft_qos"),
                jain_fairness=cf.get("jain_fairness"),
                min_rate_mbps=cf.get("min_rate_mbps"),
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
        
        # Update resource metrics plotter (samples CPU/GPU automatically)
        self.resource_plotter.update(step=self.total_steps)
    
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
            return_per_step=metrics.return_per_step,
            return_per_agent_step=metrics.return_per_agent_step,
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
            mean_soft_qos=metrics.mean_soft_qos,
            min_soft_qos=metrics.min_soft_qos,
            jain_fairness=metrics.jain_fairness,
            min_rate_mbps=metrics.min_rate_mbps,
            max_rate_mbps=metrics.max_rate_mbps,
            rate_p5_mbps=metrics.rate_p5_mbps,
            rate_p95_mbps=metrics.rate_p95_mbps,
            sinr_db_mean=metrics.sinr_db_mean,
            interference_mean_mw=metrics.interference_mean_mw,
            ap_load_mean=metrics.ap_load_mean,
        )
        
        # Console output
        print(
            f"Ep: {episode} | Steps: {metrics.steps} | "
            f"Dur: {duration:.2f}s | Return: {metrics.return_total:.3f} | "
            f"Return/Agent/Step: {metrics.return_per_agent_step:.3f} | "
            f"AvgReward: {metrics.avg_reward:.3f} | "
            f"QoSRate: {metrics.qos_rate:.3f} | SoftQoS: {metrics.mean_soft_qos:.3f} | "
            f"Jain: {metrics.jain_fairness:.3f} | Eps: {eps:.4f}"
        )
        
        # Episode-mode plotting
        if self.config.plot_x_axis == "episodes":
            self.plotter.update(
                step=episode,
                epsilon=eps,
                mean_reward=metrics.avg_reward,
                qos=metrics.qos_rate,
                capacity_sum_mbps=metrics.avg_capacity,
                episode_return=metrics.return_per_agent_step,
                mean_soft_qos=metrics.mean_soft_qos,
                jain_fairness=metrics.jain_fairness,
                min_rate_mbps=metrics.min_rate_mbps,
            )
    
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

        cellfree_diagnostics = None
        if hasattr(self.scenario, "get_visual_diagnostics"):
            cellfree_diagnostics = self.scenario.get_visual_diagnostics()
        
        self.telecom_plotter.update(
            step=episode,  # Use episode as the counter since we update per episode
            episode=episode,
            actions=actions,
            interference_edges=interference_edges,
            cellfree_diagnostics=cellfree_diagnostics,
        )
    
    def _finalize(self):
        """Clean up and write final outputs."""
        total_duration = time.time() - self.start_time
        print(f"Total training time: {total_duration/60:.2f} min ({total_duration:.2f} s)")
        
        for plotter in (self.plotter, self.network_plotter, self.telecom_plotter, self.resource_plotter):
            try:
                plotter.close()
            except Exception as exc:
                self.plot_errors.append(f"{type(plotter).__name__}: {exc}")
        self.ep_logger.close()
        self.step_logger.close()
        self.time_logger.close()
        
        # Gather feature flags
        feature_flags = {
            "gnn_enabled": self.gnn_manager.enabled if self.gnn_manager else False,
            "mobility_enabled": self.mobility_manager is not None,
            "gnn_observation_mode": getattr(self.gnn_manager, 'observation_mode', None) if self.gnn_manager else None,
            "deferred_plotting": self.opt.deferred_plotting,
        }
        
        # Add resource usage summary
        resource_summary = self.resource_plotter.get_summary()
        
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
            resource_usage=resource_summary,
        )
        


class ExperimentManager:
    """Manages experiment setup and execution."""
    
    def __init__(self, opt, sce, run_name: Optional[str] = None):
        self.opt = opt
        self.sce = sce
        self.run_name = run_name
        # Defer full GPU setup to setup(); provide a basic default for
        # code that reads self.device before setup() is called.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_gpu_device_ids: List[int] = []
        self._gpu_manager: Optional[GPUManager] = None
        
        self.scenario = None
        self.run_dir = None
        self.agents = None
        self.config = None
        self.mobility_manager = None
        self.gnn_manager = None

    def _configure_device_plan(self):
        """Auto-select single vs multi-GPU via GPUManager."""
        self._gpu_manager = GPUManager.from_opt(self.opt)
        self.device = self._gpu_manager.device
        self.multi_gpu_device_ids = self._gpu_manager.multi_gpu_device_ids
        self._gpu_manager.apply_to_opt(self.opt)

    def _auto_scale_batch_size(self, gnn_input_dim: Optional[int] = None):
        """Delegate batch-size auto-scaling to GPUManager."""
        if self._gpu_manager is None:
            return
        state_dim = gnn_input_dim if gnn_input_dim else get_flat_state_dim(self.opt, self.scenario)
        new_bs = self._gpu_manager.auto_scale_batch_size(
            current_batch_size=int(self.opt.batch_size),
            state_dim=state_dim,
        )
        self.opt["batch_size"] = new_bs
    
    def setup(self):
        """Initialize all components for a trial."""
        resolve_configuration(self.opt, self.sce)
        resolve_seeds(self.opt)
        self._configure_device_plan()
        self.scenario = create_scenario(self.sce, self.opt)
        if self.run_name:
            self.run_dir = RunDirectoryManager(prefix=self.run_name, overwrite=False)
        else:
            self.run_dir = RunDirectoryManager(overwrite=False)
        self.config = TrainingConfig.from_opt(self.opt)
        
        
        # Create mobility manager if enabled
        if self.opt.mobility_enabled:
            self.mobility_manager = MobilityManager(
                scenario=self.scenario,
                num_ues=self.opt.nagents,
                seed=self.opt.mobility_seed,
                enabled=True,
                speed=getattr(self.opt, "mobility_speed", None),
                hotspots_enabled=getattr(self.opt, "hotspots_enabled", None),
            )
        else:
            self.mobility_manager = None
        
        # Create GNN manager if enabled (reads params from opt when available)
        _gnn_enabled = self.opt.gnn_enabled
        self.gnn_manager = GNNObservationManager(
            scenario=self.scenario,
            device=self.device,
            enabled=_gnn_enabled,
            opt=self.opt,
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
            multi_gpu_device_ids=self.multi_gpu_device_ids,
        )
        
        if self.gnn_manager.enabled:
            print(f"GNN enabled: input_dim={gnn_input_dim}")

        print(f"Environment type: {resolve_environment_type(self.opt)}")

        self._auto_scale_batch_size(gnn_input_dim)
        self.opt.batch_size = min(self.opt.batch_size, self.opt.capacity)
        save_config(self.run_dir, self.opt, self.sce)
        save_environment_snapshot(self.run_dir)
    
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