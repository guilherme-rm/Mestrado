"""Evaluate saved HetNet policies with frozen weights and greedy action selection."""
from pathlib import Path
import csv
import json
import random
import time
import numpy as np
import torch
from dotdic import DotDic
from rl.checkpoints import load_checkpoint, bare
from rl.observations import enriched_observations


def restore_policy(path, device='cpu'):
    data = load_checkpoint(path)
    from telecom.scenario import Scenario
    from rl.training import create_agents, GNNObservationManager
    import rl.training as training
    opt, sce = DotDic(data['opt']), DotDic(data['sce'])
    opt.use_amp = False
    opt.multi_gpu_auto = False
    device = torch.device(device)
    scenario = Scenario(sce)
    stations = scenario.Get_BaseStations()
    if len(stations) != len(data['base_stations']):
        raise ValueError('Checkpoint base-station count is inconsistent')
    for bs, saved in zip(stations, data['base_stations']):
        if bs.bs_type != saved['type']: raise ValueError('Base-station ordering changed')
        bs._loc = saved['position'].numpy().copy()
        bs.radius = saved['radius']
    # Build graph caches only after restoring geometry.
    manager = GNNObservationManager(scenario, device, enabled=bool(opt.gnn_enabled), opt=opt)
    old_shared = training.SHARED_AGENT_NETWORKS
    training.SHARED_AGENT_NETWORKS = data['shared_networks']
    try:
        agents = create_agents(opt, sce, scenario, device,
                               gnn_input_dim=manager.output_dim if manager.enabled else None)
    finally:
        training.SHARED_AGENT_NETWORKS = old_shared
    for agent, weights, position in zip(agents, data['policies'], data['ue_positions']):
        bare(agent.model_policy).load_state_dict(weights, strict=True)
        bare(agent.model_target).load_state_dict(weights, strict=True)
        agent._location_manager._location = position.to(device).clone()
        agent.set_eval_mode()
        agent.model_policy.requires_grad_(False)
        agent.model_target.requires_grad_(False)
    if manager.enabled:
        if data['encoder'] is None: raise ValueError('Missing graph encoder weights')
        manager._encoder.load_state_dict(data['encoder'], strict=True)
        manager._encoder.eval()
        manager._encoder.requires_grad_(False)
    return agents, scenario, manager, opt, sce, data


def evaluate_checkpoint(path, output, episodes=1, steps=None, device='cpu', seed=42):
    if episodes < 1 or (steps is not None and steps < 1):
        raise ValueError('Episodes and steps must be positive')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agents, scenario, manager, opt, sce, data = restore_policy(path, device)
    from rl.training import initialize_episode, compute_rewards_and_next_state
    horizon = steps if steps is not None else int(opt.nsteps)
    if horizon < 1: raise ValueError('Checkpoint horizon must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fields = ['episode','steps','avg_reward','qos_rate','final_qos','full_qos_reached','duration_seconds']
    all_rows = []
    # No call to Save_Transition, Optimize_Model, Target_Update, or action exploration.
    with torch.inference_mode(), (output/'actions.csv').open('w') as f:
        actions_log = csv.writer(f); actions_log.writerow(['episode','step','actions'])
        for episode in range(episodes):
            ctx = initialize_episode(len(agents), torch.device(device))
            previous = None; capacity = None; rewards_sum = qos_sum = 0.; success = False
            started = time.perf_counter()
            for step in range(horizon):
                obs = manager.encode(agents, previous) if manager.enabled else enriched_observations(agents, scenario, ctx.state, previous, capacity)
                actions = torch.stack([a.model_policy(obs[i] if obs is not None else ctx.state).argmax().reshape(())
                                       for i,a in enumerate(agents)])
                rewards, qos, state, capacity, _ = compute_rewards_and_next_state(agents, actions, ctx.state, scenario)
                rewards_sum += rewards.mean().item(); qos_sum += qos.mean().item()
                full = bool(qos.eq(1).all()); success |= full
                actions_log.writerow([episode,step,json.dumps(actions.cpu().tolist())])
                ctx.state = state; previous = actions
                if bool(opt.early_termination) and full: break
            row = dict(episode=episode,steps=step+1,avg_reward=rewards_sum/(step+1),
                       qos_rate=qos_sum/(step+1),final_qos=qos.mean().item(),
                       full_qos_reached=success,duration_seconds=time.perf_counter()-started)
            all_rows.append(row)
    with (output/'evaluation_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
    metadata=dict(mode='evaluation',checkpoint=str(Path(path).resolve()),
                  checkpoint_episode=data['episode'],policy='greedy',learning=False,
                  dropout=False,mobility=False,geometry='saved checkpoint positions',seed=seed,
                  episodes=episodes,max_steps=horizon,device=str(device),
                  total_steps=sum(r['steps'] for r in all_rows))
    (output/'evaluation.json').write_text(json.dumps(metadata,indent=2))
    print(f'Evaluation complete (no training). Results: {output.resolve()}')
    return all_rows
