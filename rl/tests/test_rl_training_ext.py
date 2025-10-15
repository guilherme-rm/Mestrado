import torch
from dotdic import DotDic
from rl.training import select_actions, compute_rewards_and_next_state, create_agents, compute_epsilon
from telecom.scenario import Scenario


def make_cfg():
    opt = DotDic({
        'nagents': 2,
        'capacity': 30,
        'learningrate': 1e-3,
        'momentum': 0.0,
        'batch_size': 2,
        'gamma': 0.95,
        'nupdate': 2,
        'eps_min': 0.1,
        'eps_increment': 0.01,
        'eps_max': 0.9,
    })
    sce = DotDic({
        'nMBS': 1,
        'nPBS': 0,
        'nFBS': 0,
        'nChannel': 2,
        'rMBS': 200,
        'rPBS': 100,
        'rFBS': 50,
        'N0': -100,
        'BW': 1e6,
        'QoS_thr': 0,
        'negative_cost': -0.5,
    })
    return opt, sce


def test_select_actions_and_rewards():
    opt, sce = make_cfg()
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, torch.device('cpu'))
    state = torch.zeros(opt.nagents)
    eps = compute_epsilon(opt, 0, 0)
    actions = select_actions(agents, state, scenario, eps)
    assert actions.shape == (opt.nagents,)
    rewards, qos, next_state, capacity = compute_rewards_and_next_state(agents, actions, state, scenario)
    assert rewards.shape == (opt.nagents,)
    assert qos.shape == (opt.nagents,)
    assert next_state.shape == (opt.nagents,)
    assert capacity.shape == (opt.nagents,)
