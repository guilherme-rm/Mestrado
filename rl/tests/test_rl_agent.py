import torch
from dotdic import DotDic
from rl.agent import Agent
from rl.training import create_agents
from telecom.scenario import Scenario


def make_configs():
    opt = DotDic(
        {
            "nagents": 2,
            "capacity": 20,
            "learningrate": 1e-3,
            "momentum": 0.0,
            "batch_size": 4,
            "gamma": 0.9,
            "nupdate": 2,
        }
    )
    sce = DotDic(
        {
            "nMBS": 1,
            "nPBS": 0,
            "nFBS": 0,
            "nChannel": 2,
            "rMBS": 200,
            "rPBS": 100,
            "rFBS": 50,
            "N0": -100,
            "BW": 1e6,
            "QoS_thr": 0,
            "negative_cost": -0.5,
        }
    )
    return opt, sce


def test_agent_action_selection_shape():
    opt, sce = make_configs()
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, torch.device("cpu"))
    state = torch.zeros(opt.nagents)
    a = agents[0].Select_Action(state, scenario, eps=0.0)
    assert a.shape == (1, 1)
    assert 0 <= int(a.item()) < scenario.BS_Number() * sce.nChannel


def test_agent_reward_tuple():
    opt, sce = make_configs()
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, torch.device("cpu"))
    state = torch.zeros(opt.nagents)
    actions = torch.stack(
        [ag.Select_Action(state, scenario, eps=1.0) for ag in agents]
    ).squeeze(-1)
    qos, reward, capacity = agents[0].Get_Reward(actions, actions[0], state, scenario)
    assert qos in (0, 1)
    assert isinstance(reward.item(), float)
    assert capacity.item() >= 0.0
