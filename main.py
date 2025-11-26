"""Entry point for running wireless resource allocation training."""

import argparse
import copy
import json
import os
import time

import torch
from collections import deque

# Assuming standard project structure and dependencies are installed
# Note: Adjust import paths based on your actual file locations
from telecom.scenario import Scenario
from rl.training import (
    compute_epsilon,
    select_actions,
    compute_rewards_and_next_state,
    store_and_learn,
    TrainingContext,
    create_agents,
    initialize_episode,
    should_terminate,
)
from rl.agent import Agent
# Assuming CODE_2 is saved as functions/live_plot.py
from functions.live_plot import RealTimeStepPlotter
# Assuming CODE_7 is saved as functions/logging.py
# If CODE_7 was in functions/__init__.py, adjust the import accordingly.
from functions.logging import (
    RunDirectoryManager,
    save_config,
    save_environment_snapshot,
    EpisodeMetricsLogger,
    StepMetricsLogger,
    checkpoint_agents,
    write_summary,
)
from dotdic import DotDic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_episodes(opt, sce, agents, scenario, run_dir: RunDirectoryManager):
    """Run training episodes using modular helper functions."""

    # Configuration setup
    enable_plot = getattr(opt, 'enable_plot', True)
    plot_interval = getattr(opt, 'plot_interval', 10)
    smooth_window = getattr(opt, 'plot_smooth_window', 50)
    # To observe learning trends, plotting over episodes is generally clearer.
    x_axis_mode = getattr(opt, 'plot_x_axis', 'episodes')
    early_termination = getattr(opt, 'early_termination', True)

    state_target = torch.ones(opt.nagents, device=device)  # QoS requirement (all satisfied)
    action_tail_len = 10 # Track last N actions for debugging

    # Initialize Plotter
    plot_path = run_dir.subpath('training_progress.png')
    plotter = RealTimeStepPlotter(
        enabled=enable_plot,
        plot_interval=plot_interval,
        out_path=str(plot_path),
        smooth_window=smooth_window,
        x_axis_mode=x_axis_mode,
    )

    # Initialize Loggers
    ep_logger = EpisodeMetricsLogger(run_dir)
    # Log steps less frequently if configured
    step_logger = StepMetricsLogger(run_dir, throttle=getattr(opt, 'step_log_throttle', 1))
    checkpoint_interval = getattr(opt, 'checkpoint_interval', 100)

    # Metrics tracking (for final summary)
    episode_metrics = {
        "avg_reward": [],
        "return": [],
        "qos_rate": [],
    }

    # Timing setup
    total_start_time = time.time()
    total_steps = 0 # Track total steps globally for epsilon decay

    for episode in range(opt.nepisodes):
        ep_start = time.time()

        # Initialize episode context
        ctx = initialize_episode(opt.nagents, device)
        last_actions = [deque(maxlen=action_tail_len) for _ in range(opt.nagents)]

        step = 0
        # Metrics accumulators for the episode
        reward_sum_episode = 0.0 # Sum of mean rewards per step (for average calculation)
        return_episode = 0.0     # Sum of all rewards (Return)
        qos_sum_episode = 0.0    # Sum of mean QoS rates per step
        capacity_sum_episode = 0.0 # Sum of system capacities per step

        while step < opt.nsteps:
            # 1. Compute Epsilon based on total steps (exponential decay)
            eps = compute_epsilon(opt, total_steps)

            # 2. Select Actions
            ctx.actions = select_actions(agents, ctx.state, scenario, eps)

            # Record actions (for debugging)
            if action_tail_len > 0:
                for i in range(opt.nagents):
                    try:
                        last_actions[i].append(int(ctx.actions[i].item()))
                    except Exception:
                        last_actions[i].append(int(ctx.actions[i]))

            # 3. Environment Step (Compute Rewards and Next State)
            ctx.rewards, ctx.qos, ctx.next_state, capacity = compute_rewards_and_next_state(
                agents, ctx.actions, ctx.state, scenario
            )

            # 4. Store Experience and Learn (includes Soft Target Update)
            store_and_learn(
                agents,
                ctx.state,
                ctx.actions,
                ctx.next_state,
                ctx.rewards,
                scenario,
                step,
                opt,
            )

            # 5. Metrics accumulation
            mean_reward_step = float(ctx.rewards.mean().item())
            qos_mean_step = float(ctx.qos.mean().item())
            cap_sum_step = float(capacity.sum().item())

            reward_sum_episode += mean_reward_step
            return_episode += float(ctx.rewards.sum().item())
            qos_sum_episode += qos_mean_step
            capacity_sum_episode += cap_sum_step

            # Step Logging
            step_logger.log(
                global_step=total_steps,
                episode=episode,
                step_in_episode=step,
                epsilon=eps,
                mean_reward=mean_reward_step,
                qos_mean=qos_mean_step,
                capacity_sum_mbps=cap_sum_step,
            )

            # Plotting (if in 'steps' mode)
            if x_axis_mode == 'steps':
                plotter.update(
                    step=total_steps,
                    epsilon=eps,
                    mean_reward=mean_reward_step,
                    qos=qos_mean_step,
                    capacity_sum_mbps=cap_sum_step,
                    # Return is typically plotted per episode
                )

            # 6. Transition State
            ctx.state = ctx.next_state.clone()

            # 7. Check Termination Condition
            if early_termination and should_terminate(ctx.state, state_target):
                break
                
            step += 1
            total_steps += 1

        # --- End of Episode ---

        # Calculate Episode Averages
        step_count = max(1, step)
        avg_reward_ep = reward_sum_episode / step_count
        qos_rate_ep = qos_sum_episode / step_count
        avg_capacity_ep = capacity_sum_episode / step_count

        episode_metrics["avg_reward"].append(avg_reward_ep)
        episode_metrics["return"].append(return_episode)
        episode_metrics["qos_rate"].append(qos_rate_ep)

        # Plotting (if in 'episodes' mode)
        if x_axis_mode == 'episodes':
            plotter.update(
                step=episode,
                epsilon=eps, # Epsilon at the end of the episode
                mean_reward=avg_reward_ep,
                qos=qos_rate_ep,
                capacity_sum_mbps=avg_capacity_ep,
                episode_return=return_episode,
            )

        ep_duration = time.time() - ep_start

        # Episode CSV logging
        ep_logger.log(
            episode=episode,
            steps=step,
            avg_reward=avg_reward_ep,
            return_val=return_episode,
            qos_rate=qos_rate_ep,
            capacity_mean=avg_capacity_ep,
            epsilon_last=eps,
            duration_seconds=ep_duration,
        )

        # Console output
        print(
            f"Ep: {episode} | Steps: {step} | Dur: {ep_duration:.2f}s | Return: {return_episode:.3f} | AvgReward: {avg_reward_ep:.3f} | QoSRate: {qos_rate_ep:.3f} | Eps: {eps:.4f}"
        )

        # Checkpointing
        if checkpoint_interval and (episode + 1) % checkpoint_interval == 0:
            checkpoint_agents(run_dir, agents, episode + 1)

    # --- End of Training ---
    total_duration = time.time() - total_start_time
    print(f"Total training time: {total_duration/60:.2f} min ({total_duration:.2f} s)")
    plotter.close()
    ep_logger.close()
    step_logger.close()

    # Write final summary
    def mean_last_k(data, k=100):
        if not data: return None
        return float(sum(data[-k:]) / max(1, len(data[-k:])))

    summary = {
        'episodes': opt.nepisodes,
        'total_steps': total_steps,
        'total_training_seconds': total_duration,
        'mean_last_100_reward': mean_last_k(episode_metrics['avg_reward'], 100),
        'mean_last_100_return': mean_last_k(episode_metrics['return'], 100),
        'mean_last_100_qos': mean_last_k(episode_metrics['qos_rate'], 100),
    }
    write_summary(run_dir, summary)
    # Final checkpoint
    checkpoint_agents(run_dir, agents, opt.nepisodes, tag='final')
    return episode_metrics

def run_trial(opt, sce):
    # Initialize Scenario
    scenario = Scenario(sce)
    # Setup Run Directory (using overwrite=False to keep history of trials)
    run_dir = RunDirectoryManager(overwrite=False)
    save_config(run_dir, opt, sce)
    save_environment_snapshot(run_dir)
    # Create Agents
    agents = create_agents(opt, sce, scenario, device)
    # Start Training
    run_episodes(opt, sce, agents, scenario, run_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Using the original argument names provided by the user
    parser.add_argument('-c1', '--config_path1', type=str, required=True, help='path to scenario config (sce.json)')
    parser.add_argument('-c2', '--config_path2', type=str, required=True, help='path to RL config (opt.json)')
    parser.add_argument('-n', '--ntrials', type=int, default=1, help='number of trials to run')
    args = parser.parse_args()

    # Load configurations
    try:
        with open(args.config_path1, 'r') as f:
             sce = DotDic(json.load(f))
        with open(args.config_path2, 'r') as f:
            opt = DotDic(json.load(f))
    except FileNotFoundError as e:
        print(f"Error loading configuration file: {e}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON configuration: {e}")
        exit(1)

    for i in range(args.ntrials):
        print(f"--- Starting Trial {i+1}/{args.ntrials} ---")
        trial_opt = copy.deepcopy(opt)
        trial_sce = copy.deepcopy(sce)
        run_trial(trial_opt, trial_sce)
