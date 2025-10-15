"""Entry point for running wireless resource allocation training."""

import argparse
import copy
import json
import os
import time

import torch
from collections import deque

from telecom.scenario import Scenario
from rl.training import (
    compute_epsilon,
    select_actions,
    compute_rewards_and_next_state,
    store_and_learn,
    TrainingContext,
    create_agents,
)
from rl.agent import Agent  # noqa: F401 (kept for legacy references)
from functions.live_plot import RealTimeStepPlotter
from functions import (
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

## create_agents now resides in rl.training (imported above). Kept legacy helper removed.
    
def run_episodes(opt, sce, agents, scenario, run_dir: RunDirectoryManager):
    """Run training episodes using modular helper functions.

    Responsibilities:
      - Manage episode/step loops
      - Orchestrate action selection, reward calculation, learning
      - Log per-episode step counts
      - Track simple metrics (avg reward, qos satisfaction)
    """
    # Backward compatible optional config fields
    enable_plot = getattr(opt, 'enable_plot', True)
    plot_interval = getattr(opt, 'plot_interval', 50)
    smooth_window = getattr(opt, 'plot_smooth_window', 100)

    state_target = torch.ones(opt.nagents, device=device)  # QoS requirement (all satisfied)
    # Track last N actions per agent within an episode 
    action_tail_len =10

    # Plot output path inside run directory
    plot_path = run_dir.subpath('training_progress.png')
    x_axis_mode = getattr(opt, 'plot_x_axis', 'steps')  # 'steps' or 'episodes'
    plotter = RealTimeStepPlotter(
        enabled=enable_plot,
        plot_interval=plot_interval,
        out_path=str(plot_path),
        smooth_window=smooth_window,
        x_axis_mode=x_axis_mode,
    )
    # Structured loggers
    ep_logger = EpisodeMetricsLogger(run_dir)
    step_logger = StepMetricsLogger(run_dir, throttle=getattr(opt, 'step_log_throttle', 1))
    checkpoint_interval = getattr(opt, 'checkpoint_interval', 0)  # 0 disables
    episode_metrics = {
        "avg_reward": [],
        "qos_rate": [],
    }

    # Timing setup
    total_start_time = time.time()
    episode_time_path = run_dir.subpath('episode_times.csv')
    with open(episode_time_path, 'w', encoding='utf-8') as ftime:
        ftime.write('episode,steps,duration_seconds\n')

    for episode in range(opt.nepisodes):
        ep_start = time.time()
        # Initialize training context tensors once per episode
        ctx = TrainingContext(
            state=torch.zeros(opt.nagents, device=device),
            next_state=torch.zeros(opt.nagents, device=device),
            actions=torch.zeros(opt.nagents, dtype=torch.long, device=device),
            rewards=torch.zeros(opt.nagents, device=device),
            qos=torch.zeros(opt.nagents, device=device),
        )
        # Per-episode rolling buffers for last actions
        last_actions = [deque(maxlen=action_tail_len) for _ in range(opt.nagents)]

        step = 0
        cumulative_capacity_episode = 0.0
        step_count_episode = 0
        last_cap_sum = 0.0
        # Accumulate per-step means to compute true per-episode averages
        reward_sum_episode = 0.0
        qos_sum_episode = 0.0
        # Initialize global cumulative steps attribute once
        if episode == 0 and not hasattr(run_episodes, '_cumulative_steps'):
            run_episodes._cumulative_steps = 0  # type: ignore[attr-defined]

        while step < opt.nsteps:
            eps = compute_epsilon(opt, episode, step)
            ctx.actions = select_actions(agents, ctx.state, scenario, eps)
            # Record actions for tail printout
            if action_tail_len > 0:
                for i in range(opt.nagents):
                    try:
                        last_actions[i].append(int(ctx.actions[i].item()))
                    except Exception:
                        # Fallback to raw value if tensor indexing fails unexpectedly
                        last_actions[i].append(int(ctx.actions[i]))
            ctx.rewards, ctx.qos, ctx.next_state, capacity = compute_rewards_and_next_state(
                agents, ctx.actions, ctx.state, scenario
            )
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

            mean_reward_step = float(ctx.rewards.mean().item())
            qos_mean_step = float(ctx.qos.mean().item())
            reward_sum_episode += mean_reward_step
            qos_sum_episode += qos_mean_step

            cap_sum = float(capacity.sum().item())
            last_cap_sum = cap_sum
            if x_axis_mode == 'steps':
                # Plot per step (global cumulative step)
                plotter.update(
                    step=getattr(run_episodes, '_cumulative_steps'),
                    epsilon=eps,
                    mean_reward=mean_reward_step,
                    qos=qos_mean_step,
                    capacity_sum_mbps=cap_sum,
                )
                # Step CSV logging (global perspective)
                step_logger.log(
                    global_step=getattr(run_episodes, '_cumulative_steps'),
                    episode=episode,
                    step_in_episode=step,
                    epsilon=eps,
                    mean_reward=mean_reward_step,
                    qos_mean=qos_mean_step,
                    capacity_sum_mbps=cap_sum,
                )
                step_count_episode += 1  # count step for episode averages
            else:
                cumulative_capacity_episode += cap_sum
                step_count_episode += 1

            # Transition
            ctx.state = ctx.next_state.clone()
            # Early termination if all QoS satisfied
            if torch.all(ctx.state.eq(state_target)):
                break
            step += 1
            run_episodes._cumulative_steps += 1  # type: ignore[attr-defined]

        # Metrics tracking — true per-episode averages over steps
        avg_reward_ep = reward_sum_episode / max(1, step_count_episode)
        qos_rate_ep = qos_sum_episode / max(1, step_count_episode)
        episode_metrics["avg_reward"].append(avg_reward_ep)
        episode_metrics["qos_rate"].append(qos_rate_ep)
        if x_axis_mode == 'episodes':
            mean_capacity_episode = cumulative_capacity_episode / max(step_count_episode, 1)
            plotter.update(
                step=episode,
                epsilon=eps,
                mean_reward=avg_reward_ep,
                qos=qos_rate_ep,
                capacity_sum_mbps=mean_capacity_episode,
            )

        ep_duration = time.time() - ep_start
        # Append timing line
        with open(episode_time_path, 'a', encoding='utf-8') as ftime:
            ftime.write(f"{episode},{step},{ep_duration:.4f}\n")
        # Episode CSV logging
        cap_mean = (cumulative_capacity_episode / max(step_count_episode, 1)) if step_count_episode else last_cap_sum
        ep_logger.log(
            episode=episode,
            steps=step,
            avg_reward=avg_reward_ep,
            qos_rate=qos_rate_ep,
            capacity_mean=cap_mean,
            epsilon_last=eps,
            duration_seconds=ep_duration,
        )
        # Print last N actions per agent for this episode
        if action_tail_len > 0:
            for i in range(opt.nagents):
                print(f"Episode {episode} | Agent {i} last {action_tail_len} actions: {list(last_actions[i])}")
        # Checkpoint if interval set
        if checkpoint_interval and (episode + 1) % checkpoint_interval == 0:
            checkpoint_agents(run_dir, agents, episode + 1)
        print(
            f"Episode Number: {episode} | Steps: {step} | Duration: {ep_duration:.2f}s | AvgReward: {episode_metrics['avg_reward'][-1]:.3f} | QoSRate: {episode_metrics['qos_rate'][-1]:.3f}"
        )

    total_duration = time.time() - total_start_time
    print(f"Total training time: {total_duration/60:.2f} min ({total_duration:.2f} s)")
    plotter.close()
    ep_logger.close()
    step_logger.close()
    # Write final summary
    summary = {
        'episodes': opt.nepisodes,
        'total_training_seconds': total_duration,
        'final_avg_reward': episode_metrics['avg_reward'][-1] if episode_metrics['avg_reward'] else None,
        'final_qos_rate': episode_metrics['qos_rate'][-1] if episode_metrics['qos_rate'] else None,
        'mean_last_10_reward': float(sum(episode_metrics['avg_reward'][-10:]) / max(1, len(episode_metrics['avg_reward'][-10:]))),
        'mean_last_10_qos': float(sum(episode_metrics['qos_rate'][-10:]) / max(1, len(episode_metrics['qos_rate'][-10:]))),
    }
    write_summary(run_dir, summary)
    # Final checkpoint
    checkpoint_agents(run_dir, agents, opt.nepisodes, tag='final')
    return episode_metrics

def run_trial(opt, sce):
    scenario = Scenario(sce)
    # Create a fixed results directory and overwrite any previous contents
    run_dir = RunDirectoryManager(overwrite=True)
    save_config(run_dir, opt, sce)
    save_environment_snapshot(run_dir)
    agents = create_agents(opt, sce, scenario, device)
    run_episodes(opt, sce, agents, scenario, run_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c1', '--config_path1', type=str, help='path to existing scenarios file')
    parser.add_argument('-c2', '--config_path2', type=str, help='path to existing options file')
    parser.add_argument('-n', '--ntrials', type=int, default=1, help='number of trials to run')
    args = parser.parse_args()
    sce = DotDic(json.loads(open(args.config_path1, 'r').read()))
    opt = DotDic(json.loads(open(args.config_path2, 'r').read()))  # Load the configuration file as arguments
    for _ in range(args.ntrials):
        trial_opt = copy.deepcopy(opt)  # Deep copy in case of mutation
        trial_sce = copy.deepcopy(sce)
        run_trial(trial_opt, trial_sce)
