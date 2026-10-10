"""Regression tests for configuration, reproducibility, and complete run lifecycles."""
import copy
import json
import random
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from dotdic import DotDic
from functions.configuration import resolve_configuration
from functions.runtime_settings import apply_runtime_settings
from functions.logging import RunDirectoryManager
from functions.main_classes import ExperimentManager, Trainer
from functions.gpu_manager import GPUManager
from rl.agent import Agent, RewardCalculator
from rl.memory import ReplayMemory, TensorReplayMemory
from rl.training import create_agents, GNNObservationManager, store_and_learn
from rl.interaction import observations, greedy_actions, move_users, compute_rewards_and_next_state
from rl.evaluation import evaluate_checkpoint, restore_policy
from telecom.scenario import Scenario
from telecom.mobility import MobilityManager
from telecom_cellfree.association_manager import UserCentricAssociationManager

ROOT = Path(__file__).resolve().parents[1]


def configs():
    opt = DotDic(json.loads((ROOT/'Config/opt/config_opt_small.json').read_text()))
    opt.update(nagents=3, nepisodes=2, nsteps=4, capacity=32, batch_size=2,
               min_memory_for_learning=2, device='cpu', use_amp=False,
               enable_plot=False, checkpoint_interval=0, early_termination=False,
               dnn_hidden_layers=[8, 8], seed=42, parallel_agent_optimization=False)
    sce = DotDic(dict(nMBS=1, nPBS=1, nFBS=1, rMBS=200, rPBS=100,
                     rFBS=30, BW=1e6, nChannel=2, N0=-100, QoS_thr=0,
                     negative_cost=-.5, profit=.5, power_cost=.0005, action_cost=.001))
    return opt, sce


def test_missing_defaults_and_isolated_runtime():
    import constants
    from telecom import mobility
    missing = DotDic()
    assert getattr(missing, 'absent', 123) == 123
    assert not hasattr(missing, 'absent')
    with pytest.raises(AttributeError):
        _ = missing.absent
    before = (constants.GNN_ENABLED, mobility.MOBILITY_ENABLED)
    on = DotDic(gnn_enabled=False, gnn_transformer_enabled=True, mobility_enabled=True)
    apply_runtime_settings(on, DotDic())
    off = DotDic(gnn_enabled=False, mobility_enabled=False)
    apply_runtime_settings(off, DotDic())
    assert on.gnn_enabled and on.mobility_enabled
    assert not off.gnn_enabled and not off.mobility_enabled
    assert (constants.GNN_ENABLED, mobility.MOBILITY_ENABLED) == before
    opt, sce = configs()
    del opt['step_log_throttle']
    resolve_configuration(opt, sce)
    assert opt.step_log_throttle > 0


@pytest.mark.parametrize('key,value', [('learn_every',0), ('nsteps',0), ('batch_size',1000),
                                        ('gamma',float('nan')), ('learning_rate',0),
                                        ('architecture','unknown'), ('replay_sampling','unknown')])
def test_invalid_configuration(key, value):
    opt, sce = configs(); opt[key] = value
    with pytest.raises(ValueError): resolve_configuration(opt, sce)


def test_zero_reward_coefficients():
    opt, sce = configs(); sce.update(profit=0., power_cost=0., action_cost=0.)
    scenario = Scenario(sce)
    calculator = RewardCalculator(sce, opt, torch.device('cpu'))
    location = torch.tensor([500., 500.])
    _, reward, capacity = calculator.compute(torch.tensor([0]), torch.tensor(0), location, scenario)
    assert capacity > 0 and reward == 0


def test_output_collision_and_geometry_reads(tmp_path, monkeypatch):
    monkeypatch.setattr('functions.logging.time.strftime', lambda *a: 'same-second')
    a, b = RunDirectoryManager(str(tmp_path)), RunDirectoryManager(str(tmp_path))
    assert a.path != b.path
    _, sce = configs(); scenario = Scenario(sce, seed=42)
    before = scenario.rng.getstate()
    first = scenario.BS_Location(); second = scenario.BS_Location()
    assert scenario.rng.getstate() == before
    for left, right in zip(first, second): np.testing.assert_array_equal(left, right)
    actual = np.array([b.get_location() for b in scenario.Get_BaseStations()])
    np.testing.assert_array_equal(np.concatenate(first), actual)


def test_explicit_cpu_and_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    assert GPUManager.from_opt(DotDic(cuda=-1)).device.type == 'cpu'
    with pytest.raises(ValueError): GPUManager.from_opt(DotDic(device='cuda:3'))


@pytest.mark.parametrize('buffer_cls', [ReplayMemory, TensorReplayMemory])
def test_replay_policy_and_terminal_metadata(buffer_cls):
    memory = buffer_cls(10, sampling='without_replacement')
    for i in range(10):
        memory.Push(torch.tensor([[float(i)]]), torch.tensor([[0]]), torch.zeros(1,1), torch.ones(1), i == 9)
    batch = memory.Sample(10)
    if buffer_cls is TensorReplayMemory:
        states, _, _, _, terminal = batch
        assert states.unique().numel() == 10 and terminal.sum() == 1
    else:
        assert len({x.state.item() for x in batch}) == 10
        assert sum(x.terminated for x in batch) == 1


def test_learning_cadence_keeps_all_transitions():
    opt, sce = configs(); scenario = Scenario(sce); agents = create_agents(opt,sce,scenario,torch.device('cpu'))
    calls = []
    for a in agents: a.Optimize_Model = Mock(wraps=a.Optimize_Model)
    for step in range(4):
        store_and_learn(agents, torch.zeros(3), torch.zeros(3,dtype=torch.long), torch.ones(3),
                        torch.ones(3), scenario, step, opt, optimize=step%2==0)
    for a in agents:
        assert len(a.memory) == 4 and a.Optimize_Model.call_count == 2


def test_one_ap_cluster():
    cluster = UserCentricAssociationManager(1).cluster_from_primary(0, np.array([1., .5, .2]))
    assert cluster == [0]


@pytest.mark.parametrize('mode', ['flat', 'enriched', 'mobility', 'gcn', 'transformer', 'cell_free'])
def test_complete_training_and_evaluation(tmp_path, monkeypatch, mode):
    monkeypatch.chdir(tmp_path)
    opt, sce = configs()
    opt.update(gnn_enabled=mode=='gcn', gnn_transformer_enabled=mode=='transformer',
               mobility_enabled=mode=='mobility', mobility_decisions_per_move=2,
               observation_mode='enriched' if mode in ('enriched','mobility') else 'qos',
               gnn_output_dim=8, gnn_hidden_dim=8, gnn_num_layers=2, learn_every=2)
    if mode == 'cell_free':
        sce = DotDic(json.loads((ROOT/'Config/cell_free/config_sce_test.json').read_text()))
        sce.update(nAP=4, nChannel=2, serving_aps_per_user=1, min_serving_aps_per_user=1, max_serving_aps_per_user=1)
        opt.environment_type = 'cell_free'
    experiment = ExperimentManager(opt, sce, run_name=mode)
    result = experiment.run_trial()
    assert len(result['qos_rate']) == 2
    assert all(np.isfinite(v) for v in result['avg_reward'])
    path = experiment.run_dir.path
    status = json.loads((path/'run_status.json').read_text())
    assert status['state'] == 'completed' and status['total_steps'] == 8
    saved = json.loads((path/'opt.json').read_text())
    assert saved['seed'] == 42 and saved['reproducibility_version'] == 2
    assert saved['batch_size'] == opt.batch_size
    assert all(len(a.memory)==8 for a in experiment.agents)
    assert all(a._dqn_optimizer._opt_step == 3 for a in experiment.agents)
    if mode == 'cell_free': return
    checkpoint = path/'checkpoints/final.pt'
    rows = evaluate_checkpoint(checkpoint, tmp_path/'evaluation', steps=3)
    assert rows[0]['steps'] == 3
    assert 0 <= rows[0]['qos_rate'] <= 1
    restored, network, manager, _, _, _ = restore_policy(checkpoint)
    assert all(not a.model_policy.training for a in restored)
    if mode == 'mobility':
        mover = MobilityManager(network, len(restored), seed=123, enabled=True, speed=5.)
        for i,a in enumerate(restored): mover.set_ue_position(i,a.location.cpu().numpy())
        before = torch.stack([a.location.clone() for a in restored])
        for a in restored: monkeypatch.setattr(a, 'Get_Reward', Mock(side_effect=AssertionError('movement must not evaluate QoS')))
        move_users(restored, mover)
        assert not torch.equal(before, torch.stack([a.location for a in restored]))
        assert all(len(a.memory)==0 for a in restored)


def test_reproducible_setup_and_no_runtime_leak(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    positions = []
    for mobile in (False, True, False):
        opt, sce = configs(); opt.mobility_enabled = mobile
        experiment = ExperimentManager(opt, sce); experiment.setup()
        assert (experiment.mobility_manager is not None) == mobile
        positions.append(torch.stack([a.location for a in experiment.agents]))
    torch.testing.assert_close(positions[0], positions[2])


def test_failure_flushes_logs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt, sce = configs(); experiment = ExperimentManager(opt,sce);experiment.setup()
    # Write to a real logger to exercise cleanup even before the first episode completes.
    def fail(self, episode):
        self.time_logger.log(global_step=123)
        raise RuntimeError('injected failure')
    monkeypatch.setattr(Trainer,'_run_episode',fail)
    with pytest.raises(RuntimeError,match='injected failure'): experiment.run()
    status = json.loads(experiment.run_dir.subpath('run_status.json').read_text())
    assert status['state'] == 'failed'
    assert '123' in experiment.run_dir.subpath('time_metrics.txt').read_text()


def test_plot_error_does_not_lose_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt, sce = configs(); experiment = ExperimentManager(opt,sce);experiment.setup()
    monkeypatch.setattr('functions.main_classes.RealTimeStepPlotter.close',Mock(side_effect=RuntimeError('plot failed')))
    experiment.run()
    status=json.loads(experiment.run_dir.subpath('run_status.json').read_text())
    assert status['state']=='completed' and status['plot_errors']
    assert experiment.run_dir.subpath('checkpoints/final.pt').exists()


@pytest.mark.parametrize('bootstrap,terminated,expected', [(True,True,2.8),(False,True,1.),(False,False,2.8)])
def test_bootstrap_terminal_vs_time_limit(bootstrap,terminated,expected):
    from rl.agent import DQNOptimizer
    opt=DotDic(batch_size=1,gamma=.9,bootstrap_on_termination=bootstrap,use_amp=False)
    policy=torch.nn.Linear(1,1); target=torch.nn.Linear(1,1)
    with torch.no_grad():
        policy.weight.zero_();policy.bias.zero_();target.weight.zero_();target.bias.fill_(2.)
    memory=TensorReplayMemory(2)
    memory.Push(torch.zeros(1,1),torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1),torch.ones(1),terminated)
    trainer=DQNOptimizer(policy,target,torch.optim.SGD(policy.parameters(),lr=.01),opt,torch.device('cpu'))
    result=trainer.optimize(memory)
    assert result.did_learn and result.target_q_mean == pytest.approx(expected)


def test_cellfree_snapshot_once_per_joint_action():
    from telecom_cellfree.cellfree_scenario import CellFreeScenario
    opt,sce=configs();opt.environment_type='cell_free'
    sce=DotDic(json.loads((ROOT/'Config/cell_free/config_sce_test.json').read_text()))
    sce.nAP=4
    scenario=CellFreeScenario(sce);agents=create_agents(opt,sce,scenario,torch.device('cpu'))
    calls=[]
    for agent in agents:
        calculator=agent._reward_calculator
        calculator._compute_step_snapshot=Mock(wraps=calculator._compute_step_snapshot)
        calls.append(calculator._compute_step_snapshot)
    # Same-location UEs must be addressed by index, not nearest-position lookup.
    for agent in agents: agent.set_location([100.,100.])
    for step in range(2):
        rewards,qos,state,capacity,_=compute_rewards_and_next_state(agents,torch.tensor([0,1,2]),torch.zeros(18),scenario)
        np.testing.assert_allclose(capacity.numpy(),scenario._reward_snapshot[1]['rates_mbps'],rtol=1e-6)
    assert sum(call.call_count for call in calls)==2


def test_radio_implementations_agree():
    from telecom.radio import received_power_mw
    _,sce=configs();scenario=Scenario(sce)
    for bs in scenario.Get_BaseStations():
        for distance in (0.,1.,10.,bs.radius,bs.radius+1):
            actual=received_power_mw(torch.tensor(distance),bs.transmit_power_dbm(),bs.radius,bs.bs_type=='FBS')
            assert actual.item()==pytest.approx(bs.receive_power(distance),rel=2e-6)


def test_graph_rejects_ignored_modes_and_initial_fake_channels():
    from rl.gnn.observation_encoder import GNNObservationEncoder
    from rl.gnn.graph_builder import WirelessGraphBuilder, EDGE_TYPE_INTERF
    _,sce=configs();scenario=Scenario(sce)
    with pytest.raises(ValueError,match='replace'):
        GNNObservationEncoder(scenario,torch.device('cpu'),mode='augment')
    graph=WirelessGraphBuilder(scenario,torch.device('cpu')).build([np.array([500.,500.]),np.array([510.,510.])])
    assert graph.edge_index.shape[1]>0
    assert not any(int(a)<2 and int(b)<2 for a,b in graph.edge_index.t().tolist())


def test_cli_seed_override_and_independent_trials(tmp_path):
    import subprocess,sys,os
    opt,sce=configs();opt.nepisodes=1;opt.nsteps=2
    (tmp_path/'opt.json').write_text(json.dumps(opt));(tmp_path/'sce.json').write_text(json.dumps(sce))
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    result=subprocess.run([sys.executable,str(ROOT/'main.py'),'-c1',str(tmp_path/'sce.json'),
                           '-c2',str(tmp_path/'opt.json'),'--seed','123','-n','2'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    saved=[json.loads(p.read_text()) for p in (tmp_path/'Result').glob('*/opt.json')]
    assert sorted(x['seed'] for x in saved)==[123,124]
    assert len({x['layout_seed'] for x in saved})==2
    assert len({x['mobility_seed'] for x in saved})==2


def test_version1_checkpoint_still_loads(tmp_path):
    from rl.checkpoints import save_checkpoint
    opt,sce=configs();scenario=Scenario(sce)
    agents=create_agents(opt,sce,scenario,torch.device('cpu'))
    path=tmp_path/'legacy.pt';save_checkpoint(path,agents,scenario,opt,sce,None,0,1)
    data=torch.load(path,weights_only=True);data['format_version']=1
    for key in ('observation_schema','architecture','source'):data.pop(key)
    torch.save(data,path)
    restored,_,_,_,_,_=restore_policy(path)
    for original,loaded in zip(agents,restored):
        torch.testing.assert_close(original.model_policy(torch.zeros(3)),loaded.model_policy(torch.zeros(3)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
@pytest.mark.parametrize('mode',['enriched','gcn','transformer'])
def test_cuda_training_and_checkpoint(tmp_path,monkeypatch,mode):
    monkeypatch.chdir(tmp_path)
    opt,sce=configs();opt.update(device='cuda:0',use_amp=True,nepisodes=1,
                                observation_mode='enriched' if mode=='enriched' else 'qos',
                                gnn_enabled=mode=='gcn',gnn_transformer_enabled=mode=='transformer',
                                gnn_hidden_dim=8,gnn_output_dim=8,gnn_num_layers=2)
    experiment=ExperimentManager(opt,sce);experiment.run_trial()
    assert all(a._dqn_optimizer._opt_step>0 for a in experiment.agents)
    restored,*_=restore_policy(experiment.run_dir.subpath('checkpoints/final.pt'),device='cpu')
    assert all(torch.isfinite(a.model_policy(torch.zeros(a.model_policy.input_dim))).all() for a in restored)


def test_shared_update_sees_all_agents():
    opt,sce=configs();opt.shared_agent_networks=True
    scenario=Scenario(sce);agents=create_agents(opt,sce,scenario,torch.device('cpu'))
    observed=[]
    for agent in agents:
        original=agent.Optimize_Model
        agent.Optimize_Model=lambda a=agent,f=original: (observed.append(len(a.memory)), f())[1]
    store_and_learn(agents,torch.zeros(3),torch.zeros(3,dtype=torch.long),torch.ones(3),torch.ones(3),scenario,0,opt)
    assert observed==[3]


def test_sigterm_marks_interrupted_and_flushes(tmp_path):
    import subprocess,sys,os,time
    opt,sce=configs();opt.nepisodes=100000
    (tmp_path/'opt.json').write_text(json.dumps(opt));(tmp_path/'sce.json').write_text(json.dumps(sce))
    with (tmp_path/'console.log').open('w') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'main.py'),'-c1',str(tmp_path/'sce.json'),
                                  '-c2',str(tmp_path/'opt.json')],cwd=tmp_path,stdout=log,stderr=log,
                                 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg'))
        try:
            deadline=time.monotonic()+20
            paths=[]
            while time.monotonic()<deadline and process.poll() is None:
                paths=list((tmp_path/'Result').glob('*/run_status.json'))
                if paths: break
                time.sleep(.05)
            assert paths,(tmp_path/'console.log').read_text()
            process.terminate()
            process.wait(timeout=15)
            assert json.loads(paths[0].read_text())['state']=='interrupted'
        finally:
            if process.poll() is None:
                process.kill();process.wait()
