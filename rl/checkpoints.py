"""Portable HetNet policy snapshots for inference (not exact training resume)."""
from pathlib import Path
import os
import torch
from functions.reproducibility import source_revision


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
    payload = {
        'format_version': 2, 'purpose': 'policy_evaluation',
        'source': source_revision(),
        'architecture': options.get('architecture', 'mlp'),
        'observation_schema': {'version': 1, 'mode': options.get('observation_mode', 'qos'),
                               'graph': bool(gnn_manager and gnn_manager.enabled),
                               'input_dim': bare(agents[0].model_policy).input_dim},
        'opt': options, 'sce': dict(sce),
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
    if data.get('format_version') not in (1, 2) or data.get('purpose') != 'policy_evaluation':
        raise ValueError('Unsupported policy checkpoint format')
    if data['format_version'] == 2:
        schema = data.get('observation_schema', {})
        if schema.get('version') != 1:
            raise ValueError('Unsupported observation schema')
        if data.get('architecture') != data['opt'].get('architecture', 'mlp'):
            raise ValueError('Checkpoint architecture metadata is inconsistent')
    n = data['opt']['nagents']
    if len(data['policies']) != n or len(data['ue_positions']) != n:
        raise ValueError('Checkpoint agent count is inconsistent')
    return data
