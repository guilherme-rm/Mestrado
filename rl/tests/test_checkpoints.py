"""Policy snapshots must preserve inference and never train during evaluation."""
import json
import pytest
import torch
from dotdic import DotDic
from rl.agent import Agent
from rl.training import create_agents, GNNObservationManager
from telecom.scenario import Scenario
from rl.checkpoints import save_checkpoint, cpu_state
from rl.evaluation import restore_policy, evaluate_checkpoint


@pytest.mark.parametrize('architecture', ['flat', 'gcn', 'transformer'])
def test_roundtrip_and_frozen_evaluation(tmp_path, monkeypatch, architecture):
    opt = DotDic(dict(nagents=2, capacity=20, learning_rate=0.001, momentum=0.,
                     batch_size=2, gamma=.9, nupdate=2, nsteps=3,
                     early_termination=False, dnn_hidden_layers=[8, 8],
                     mobility_enabled=True, mobility_decisions_per_move=1,
                     mobility_seed=17,
                     gnn_enabled=architecture != 'flat', gnn_conv_type=architecture,
                     gnn_output_dim=8, gnn_hidden_dim=8, gnn_num_layers=2))
    sce = DotDic(dict(nMBS=1, nPBS=1, nFBS=0, nChannel=2, rMBS=200,
                     rPBS=100, rFBS=50, N0=-100, BW=1e6, QoS_thr=0, negative_cost=-.5))
    device = torch.device('cpu')
    scenario = Scenario(sce)
    manager = GNNObservationManager(scenario, device, enabled=opt.gnn_enabled, opt=opt)
    agents = create_agents(opt, sce, scenario, device,
                           gnn_input_dim=manager.output_dim if manager.enabled else None)
    if manager.enabled:
        manager._encoder.eval()
    obs = manager.encode(agents) if manager.enabled else {i: torch.zeros(2) for i in range(2)}
    expected = [a.model_policy(obs[i]).detach().clone() for i, a in enumerate(agents)]
    path = tmp_path/'policy.pt'
    save_checkpoint(path, agents, scenario, opt, sce, manager, 2, 9)
    restored, network, encoder, _, _, data = restore_policy(path)
    actual_obs = encoder.encode(restored) if encoder.enabled else obs
    assert data['episode'] == 2
    assert data['format_version'] == 3
    assert data['observation_schema']['version'] == 2
    assert data['observation_schema']['features']
    assert data['model_spec']['architecture'] == 'mlp'
    assert data['model_spec']['input_dim'] == agents[0].model_policy.input_dim
    assert data['model_spec']['output_dim'] == agents[0].model_policy.layers[-1].out_features
    assert data['source']['commit']
    assert data['source']['dirty'] is True
    assert data['source']['working_tree_sha256']
    assert data['mobility_runtime']['enabled'] is True
    assert data['mobility_runtime']['decisions_per_move'] == 1
    for i, agent in enumerate(restored):
        assert not agent.model_policy.training
        torch.testing.assert_close(agent.location, agents[i].location)
        torch.testing.assert_close(agent.model_policy(actual_obs[i]), expected[i])
    for original, loaded in zip(scenario.Get_BaseStations(), network.Get_BaseStations()):
        assert (original.Get_Location() == loaded.Get_Location()).all()
    before = [cpu_state(a.model_policy) for a in restored]
    encoder_before = cpu_state(encoder._encoder) if encoder.enabled else None

    def forbidden(*args, **kwargs):
        raise AssertionError('Evaluation attempted a training operation')
    for method in ['Save_Transition', 'Optimize_Model', 'Target_Update']:
        monkeypatch.setattr(Agent, method, forbidden)
    monkeypatch.setattr('rl.evaluation.restore_policy', lambda *a: (restored, network, encoder, opt, sce, data))
    rows = evaluate_checkpoint(path, tmp_path/'eval', episodes=2, mobility=False)
    assert [r['steps'] for r in rows] == [3, 3]
    mobility_rows = evaluate_checkpoint(path, tmp_path/'eval_mobility', steps=3)
    assert mobility_rows[0]['steps'] == 3
    mobility_metadata = json.loads((tmp_path/'eval_mobility'/'evaluation.json').read_text())
    assert mobility_metadata['mobility'] is True
    assert mobility_metadata['mobility_seed'] == 17
    assert mobility_metadata['mobility_decisions_per_move'] == 1
    for original, agent in zip(before, restored):
        for key, value in cpu_state(agent.model_policy).items():
            assert torch.equal(value, original[key])
    if encoder.enabled:
        assert not encoder._encoder.training
        for key, value in cpu_state(encoder._encoder).items():
            assert torch.equal(value, encoder_before[key])
    lines = (tmp_path/'eval'/'actions.csv').read_text().splitlines()[1:]
    assert [x.split(',', 1)[1] for x in lines[:3]] == [x.split(',', 1)[1] for x in lines[3:]]
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(path, tmp_path/'eval')
