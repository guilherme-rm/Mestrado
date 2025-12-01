import torch
from rl.training import should_terminate, initialize_episode


class Opt:
    def __init__(self, eps_min=0.1, eps_increment=0.01, eps_max=0.9):
        self.eps_min = eps_min
        self.eps_increment = eps_increment
        self.eps_max = eps_max


def test_should_terminate():
    state = torch.ones(5)
    target = torch.ones(5)
    assert should_terminate(state, target) is True
    state2 = torch.tensor([1, 1, 0, 1, 1], dtype=torch.float32)
    assert should_terminate(state2, target) is False


def test_initialize_episode_shapes():
    ctx = initialize_episode(4, torch.device("cpu"))
    assert ctx.state.shape == (4,)
    assert ctx.actions.shape == (4,)
    assert ctx.rewards.shape == (4,)
