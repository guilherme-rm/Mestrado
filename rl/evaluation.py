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
from rl.networks import resolve_model_spec
from rl.observations import observation_schema
from rl.interaction import EnvironmentInteraction, movement_due
from rl.metrics import qos_summary


def restore_policy(path, device='cpu'):
    data = load_checkpoint(path)
    from telecom.scenario import Scenario
    from rl.training import create_agents, GNNObservationManager
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
    opt.shared_agent_networks = data['shared_networks']
    agents = create_agents(opt, sce, scenario, device,
                           gnn_input_dim=manager.output_dim if manager.enabled else None)
    if data['format_version'] in (2, 3):
        schema = data['observation_schema']
        if schema['input_dim'] != bare(agents[0].model_policy).input_dim:
            raise ValueError('Checkpoint observation dimension is inconsistent')
        if schema['mode'] != getattr(opt, 'observation_mode', 'qos') or schema['graph'] != manager.enabled:
            raise ValueError('Checkpoint observation metadata is inconsistent')
        if data['format_version'] == 3 and schema != observation_schema(
                opt, scenario, bare(agents[0].model_policy).input_dim, graph=manager.enabled):
            raise ValueError('Checkpoint observation feature schema is inconsistent')
    if 'model_spec' in data:
        restored_spec = resolve_model_spec(
            opt, sce, scenario, bare(agents[0].model_policy).input_dim).to_dict()
        if data['model_spec'] != restored_spec:
            raise ValueError('Checkpoint model dimensions are inconsistent')
    for agent, weights, position in zip(agents, data['policies'], data['ue_positions']):
        bare(agent.model_policy).load_state_dict(weights, strict=True)
        bare(agent.model_target).load_state_dict(weights, strict=True)
        agent.set_location(position)
        agent.set_eval_mode()
        agent.model_policy.requires_grad_(False)
        agent.model_target.requires_grad_(False)
    if manager.enabled:
        if data['encoder'] is None: raise ValueError('Missing graph encoder weights')
        manager._encoder.load_state_dict(data['encoder'], strict=True)
        manager._encoder.eval()
        manager._encoder.requires_grad_(False)
    return agents, scenario, manager, opt, sce, data


def evaluate_checkpoint(path, output, episodes=1, steps=None, device='cpu', seed=42,
                        mobility=None, mobility_seed=None):
    if episodes < 1 or (steps is not None and steps < 1):
        raise ValueError('Episodes and steps must be positive')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agents, scenario, manager, opt, sce, data = restore_policy(path, device)
    from rl.training import initialize_episode
    interaction = EnvironmentInteraction(agents, scenario, manager)
    horizon = steps if steps is not None else int(opt.nsteps)
    if horizon < 1: raise ValueError('Checkpoint horizon must be positive')
    mobility_config = data.get('mobility_runtime', {})
    use_mobility = bool(mobility_config.get('enabled', getattr(opt, 'mobility_enabled', False))) if mobility is None else bool(mobility)
    movement_cadence = int(mobility_config.get(
        'decisions_per_move', getattr(opt, 'mobility_decisions_per_move', 0)) or 0)
    resolved_mobility_seed = mobility_seed
    if resolved_mobility_seed is None:
        resolved_mobility_seed = mobility_config.get('seed', getattr(opt, 'mobility_seed', None))
    if resolved_mobility_seed is None:
        resolved_mobility_seed = seed
    mobility_speed = float(mobility_config.get('ue_speed_metres_per_update', 15.0))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fields = ['episode','steps','avg_reward','qos_rate','final_qos','full_qos_reached','duration_seconds']
    all_rows = []
    # No call to Save_Transition, Optimize_Model, Target_Update, or action exploration.
    with torch.inference_mode(), (output/'actions.csv').open('w') as f:
        actions_log = csv.writer(f); actions_log.writerow(['episode','step','actions'])
        for episode in range(episodes):
            mover = None
            if use_mobility:
                from telecom.mobility import MobilityManager
                mover = MobilityManager(
                    scenario, len(agents),
                    seed=(int(resolved_mobility_seed) + episode) % (2**32),
                    enabled=True, speed=mobility_speed)
                for i, agent in enumerate(agents):
                    mover.set_ue_position(i, agent.location.detach().cpu().numpy())
                interaction.move_users(mover)
            ctx = initialize_episode(len(agents), torch.device(device))
            previous = None; capacity = None; rewards_sum = qos_sum = 0.; success = False
            started = time.perf_counter()
            for step in range(horizon):
                actions = interaction.greedy_actions(ctx.state, previous, capacity)
                rewards, qos, state, capacity, _ = interaction.apply_joint_action(actions, ctx.state)
                step_qos = qos_summary(qos)
                rewards_sum += rewards.mean().item(); qos_sum += step_qos['qos_fraction']
                full = step_qos['all_satisfied']; success |= full
                actions_log.writerow([episode,step,json.dumps(actions.cpu().tolist())])
                ctx.state = state; previous = actions
                next_step = step + 1
                if mover is not None and movement_due(next_step, movement_cadence, horizon):
                    interaction.move_users(mover)
                if bool(opt.early_termination) and full: break
            row = dict(episode=episode,steps=step+1,avg_reward=rewards_sum/(step+1),
                       qos_rate=qos_sum/(step+1),final_qos=step_qos['qos_fraction'],
                       full_qos_reached=success,duration_seconds=time.perf_counter()-started)
            all_rows.append(row)
    with (output/'evaluation_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_rows)
    metadata=dict(mode='evaluation',checkpoint=str(Path(path).resolve()),
                  checkpoint_episode=data['episode'],policy='greedy',learning=False,
                  dropout=False,mobility=use_mobility,
                  mobility_seed=int(resolved_mobility_seed) if use_mobility else None,
                  mobility_decisions_per_move=movement_cadence if use_mobility else None,
                  ue_speed_metres_per_update=mobility_speed if use_mobility else None,
                  geometry='saved checkpoint positions',seed=seed,
                  episodes=episodes,max_steps=horizon,device=str(device),
                  total_steps=sum(r['steps'] for r in all_rows))
    (output/'evaluation.json').write_text(json.dumps(metadata,indent=2))
    print(f'Evaluation complete (no training). Results: {output.resolve()}')
    return all_rows
