"""Portable HetNet policy snapshots for inference (not exact training resume)."""
from pathlib import Path
import os
import torch
from functions.reproducibility import source_revision
from telecom import mobility
from .networks import resolve_model_spec
from .observations import observation_schema


def bare(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in bare(model).state_dict().items()}


def save_checkpoint(path, agents, scenario, opt, sce, gnn_manager, episode, total_steps):
    if hasattr(scenario, 'begin_step'):
        raise ValueError('Policy checkpoints currently support HetNet only')
    options = dict(opt)
    options['dnn_hidden_layers'] = [layer.out_features for layer in bare(agents[0].model_policy).layers[:-1]]
    options['gnn_enabled'] = bool(gnn_manager and gnn_manager.enabled)
    graph_enabled = bool(gnn_manager and gnn_manager.enabled)
    input_dim = bare(agents[0].model_policy).input_dim
    model_spec = resolve_model_spec(options, sce, scenario, input_dim)
    payload = {
        'format_version': 3, 'purpose': 'policy_evaluation',
        'source': source_revision(),
        'architecture': options.get('architecture', 'mlp'),
        'model_spec': model_spec.to_dict(),
        'observation_schema': observation_schema(opt, scenario, input_dim, graph=graph_enabled),
        'opt': options, 'sce': dict(sce),
        'mobility_runtime': {
            'enabled': bool(options.get('mobility_enabled', False)),
            'seed': options.get('mobility_seed'),
            'decisions_per_move': int(options.get('mobility_decisions_per_move', 0) or 0),
            'ue_speed_metres_per_update': float(mobility.UE_SPEED),
            'pause_episodes': int(mobility.UE_PAUSE_EPISODES),
            'constrain_to_coverage': bool(mobility.CONSTRAIN_TO_COVERAGE),
            'hotspots_enabled': bool(mobility.HOTSPOT_ENABLED),
            'hotspot_attraction_probability': float(mobility.HOTSPOT_ATTRACTION_PROBABILITY),
            'hotspot_count_range': list(mobility.HOTSPOT_COUNT_RANGE),
            'hotspot_radius_range': list(mobility.HOTSPOT_RADIUS_RANGE),
            'hotspot_weight_range': list(mobility.HOTSPOT_WEIGHT_RANGE),
            'hotspot_speed': float(mobility.HOTSPOT_SPEED),
            'hotspot_lifetime_range': list(mobility.HOTSPOT_LIFETIME_RANGE),
        },
        'shared_networks': bool(agents[0].uses_shared_networks),
        'episode': int(episode), 'total_steps': int(total_steps),
        'policies': [cpu_state(a.model_policy) for a in agents],
        'targets': [cpu_state(a.model_target) for a in agents],
        'ue_positions': [a.location.detach().cpu().clone() for a in agents],
        'base_stations': [{'type': b.bs_type, 'radius': float(b.radius),
                           'position': torch.as_tensor(b.Get_Location()).cpu().clone()}
                          for b in scenario.Get_BaseStations()],
        'encoder': cpu_state(gnn_manager._encoder) if options['gnn_enabled'] else None,
    }
    # Capture effective encoder defaults, not just optional input keys.
    if options['gnn_enabled']:
        from constants import (GNN_OUTPUT_DIM, GNN_HIDDEN_DIM, GNN_NUM_LAYERS,
                               GNN_USE_ATTENTION, GNN_INCLUDE_INTERFERENCE_EDGES,
                               GNN_OBSERVATION_MODE, GNN_HETEROGENEOUS)
        defaults = dict(gnn_observation_mode=GNN_OBSERVATION_MODE,
                        gnn_heterogeneous=GNN_HETEROGENEOUS, gnn_output_dim=GNN_OUTPUT_DIM, gnn_hidden_dim=GNN_HIDDEN_DIM,
                        gnn_num_layers=GNN_NUM_LAYERS, gnn_use_attention=GNN_USE_ATTENTION,
                        gnn_include_interference_edges=GNN_INCLUDE_INTERFERENCE_EDGES,
                        gnn_conv_type='gcn', gnn_transformer_enabled=False)
        for key, value in defaults.items():
            if options.get(key) is None: options[key] = value
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()
    return path


def load_checkpoint(path):
    # Only tensors and primitive containers are stored; no arbitrary object loading.
    data = torch.load(path, map_location='cpu', weights_only=True)
    if data.get('format_version') not in (1, 2, 3) or data.get('purpose') != 'policy_evaluation':
        raise ValueError('Unsupported policy checkpoint format')
    if data['format_version'] in (2, 3):
        schema = data.get('observation_schema', {})
        expected_version = 2 if data['format_version'] == 3 else 1
        if schema.get('version') != expected_version:
            raise ValueError('Unsupported observation schema')
        if data.get('architecture') != data['opt'].get('architecture', 'mlp'):
            raise ValueError('Checkpoint architecture metadata is inconsistent')
        model_spec = data.get('model_spec')
        if model_spec is not None and model_spec['architecture'] != data['architecture']:
            raise ValueError('Checkpoint model specification is inconsistent')
    n = data['opt']['nagents']
    if len(data['policies']) != n or len(data['ue_positions']) != n:
        raise ValueError('Checkpoint agent count is inconsistent')
    return data
