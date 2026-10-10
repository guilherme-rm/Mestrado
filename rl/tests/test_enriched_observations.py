import json
from pathlib import Path
import numpy as np
import pytest
import torch
from dotdic import DotDic
from telecom.scenario import Scenario
from rl.training import create_agents, get_flat_state_dim
from rl.observations import enriched_observations
from rl.checkpoints import save_checkpoint
from rl.evaluation import restore_policy, evaluate_checkpoint
from functions.main_classes import EpisodeRunner, TrainingConfig

ROOT = Path(__file__).resolve().parents[2]


def setup():
    opt = DotDic(json.loads((ROOT/'Config/opt/config_opt_small_enriched_10ue_seed42.json').read_text()))
    sce = DotDic(json.loads((ROOT/'Experiments/hetnet_thesis_45/scenarios/small_ch30.json').read_text()))
    scenario = Scenario(sce)
    opt.batch_size = 2
    opt.min_memory_for_learning = 2
    opt.capacity = 20
    agents = create_agents(opt, sce, scenario, torch.device('cpu'))
    return opt, sce, scenario, agents


def test_config_changes_observations_only():
    old = json.loads((ROOT/'Config/opt/config_opt_small_flat_10ue_seed42.json').read_text())
    new = json.loads((ROOT/'Config/opt/config_opt_small_enriched_10ue_seed42.json').read_text())
    assert new.pop('observation_mode') == 'enriched'
    assert new == old


def test_features_geometry_radio_feedback_and_legacy():
    opt, sce, scenario, agents = setup()
    qos = torch.arange(10) % 2
    b = scenario.Get_BaseStations()[0]
    agents[0]._location_manager._location = torch.tensor(b.Get_Location(), dtype=torch.float32) + torch.tensor([10., 0.])
    obs = enriched_observations(agents, scenario, qos)
    assert get_flat_state_dim(opt, scenario) == 119
    assert all(x.shape == (119,) and torch.isfinite(x).all() for x in obs.values())
    torch.testing.assert_close(obs[0][:10], qos.float())
    assert obs[0][10].item() == pytest.approx(10/b.radius, abs=1e-6)
    expected_dbm = 10*np.log10(b.receive_power(10))
    assert obs[0][36].item() == pytest.approx((np.clip(expected_dbm,-140,20)+140)/160, abs=1e-6)
    assert torch.count_nonzero(obs[0][62:]) == 0
    actions = torch.arange(10)
    # A synthetic 10dB feedback measurement, expressed as Shannon capacity.
    capacity = torch.full((10,), sce.BW*np.log2(11)/1e6)
    feedback = enriched_observations(agents, scenario, qos, actions, capacity)
    assert feedback[0][62].item() == 1  # BS0
    assert feedback[3][88+3].item() == 1  # channel3
    assert feedback[0][-1].item() == pytest.approx(.6, abs=1e-6)
    agents[0]._location_manager._location += torch.tensor([600., 0.])
    moved = enriched_observations(agents, scenario, qos, actions, capacity)
    assert moved[0][10] > 1
    assert moved[0][36] == 0  # no received power outside BS0 coverage
    torch.testing.assert_close(moved[0][62:], feedback[0][62:])
    torch.testing.assert_close(moved[1], feedback[1])
    opt.observation_mode = 'qos'
    assert enriched_observations(agents, scenario, qos) is None
    assert get_flat_state_dim(opt, scenario) == 10


def test_replay_learning_reset_and_checkpoint(tmp_path):
    opt, sce, scenario, agents = setup()
    config = TrainingConfig.from_opt(opt)
    runner = EpisodeRunner(agents, scenario, config, torch.device('cpu'))
    runner.reset()
    old_weights = agents[0].model_policy.layers[0].weight.detach().clone()
    for _ in range(3):
        runner.run_step(.2)
        expected = runner._gnn_obs[0].clone()
        runner.learn(opt)
        memory = agents[0].memory
        if hasattr(memory, '_states'):
            index = (memory._pos - 1) % memory.capacity
            torch.testing.assert_close(memory._states[index], expected)
            assert memory._next_states[index].shape == (119,)
        else:
            transition = memory.memory[-1]
            torch.testing.assert_close(transition.state.squeeze(0), expected)
            assert transition.next_state.shape == (1,119)
        runner.advance()
    assert len(agents[0].memory) == 3
    assert not torch.equal(old_weights, agents[0].model_policy.layers[0].weight)
    checkpoint = tmp_path/'enriched.pt'
    save_checkpoint(checkpoint, agents, scenario, opt, sce, None, 0, 3)
    loaded, net, _, loaded_opt, _, _ = restore_policy(checkpoint)
    assert loaded_opt.observation_mode == 'enriched'
    original_obs = enriched_observations(agents, scenario, torch.zeros(10))
    loaded_obs = enriched_observations(loaded, net, torch.zeros(10))
    for i in range(10):
        torch.testing.assert_close(original_obs[i], loaded_obs[i])
        torch.testing.assert_close(agents[i].model_policy(original_obs[i]), loaded[i].model_policy(loaded_obs[i]))
    rows = evaluate_checkpoint(checkpoint, tmp_path/'evaluation', steps=2)
    assert len(rows)==1
    runner.reset()
    runner.run_step(.2)
    assert torch.count_nonzero(runner._gnn_obs[0][62:]) == 0
