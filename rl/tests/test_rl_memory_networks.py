import torch

from rl.memory import ReplayMemory, Transition
from rl.networks import DNN
from dotdic import DotDic
from telecom.scenario import Scenario


def make_opts():
    opt = DotDic({
        'nagents': 3,
        'capacity': 10,
    })
    sce = DotDic({
        'nMBS': 1,
        'nPBS': 0,
        'nFBS': 0,
        'nChannel': 2,
        'rMBS': 100,
        'rPBS': 50,
        'rFBS': 20,
    })
    return opt, sce


def test_replay_memory_push_and_sample():
    opt, sce = make_opts()
    mem = ReplayMemory(5)
    s = torch.zeros(opt.nagents)
    a = torch.tensor([[0]])
    ns = torch.ones(opt.nagents)
    r = torch.tensor([1.0])
    for _ in range(5):
        mem.Push(s.unsqueeze(0), a, ns.unsqueeze(0), r)
    assert len(mem) == 5
    batch = mem.Sample(3)
    assert len(batch) == 3
    assert isinstance(batch[0], Transition)


def test_dnn_output_shape():
    opt, sce = make_opts()
    scenario = Scenario(sce)
    net = DNN(opt, sce, scenario)
    x = torch.zeros(opt.nagents)
    out = net(x)
    assert out.shape == (scenario.BS_Number() * sce.nChannel,)
