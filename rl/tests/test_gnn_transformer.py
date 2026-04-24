import pytest
import torch

pytest.importorskip("torch_geometric")

from dotdic import DotDic
from telecom.scenario import Scenario
from rl.gnn.observation_encoder import GNNObservationEncoder


def make_sce():
    return DotDic(
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


def test_transformer_gnn_encoder_forward():
    sce = make_sce()
    scenario = Scenario(sce)
    device = torch.device("cpu")

    encoder = GNNObservationEncoder(
        scenario=scenario,
        device=device,
        conv_type="transformer",
        gnn_output_dim=32,
        gnn_hidden_dim=32,
        gnn_num_layers=2,
        use_attention=False,
    )

    ue_positions = [torch.tensor([0.0, 0.0]), torch.tensor([20.0, 10.0])]
    actions = [0, 1]

    obs = encoder(ue_positions=ue_positions, actions=actions, ue_sinrs=[0.0, 1.0])

    assert set(obs.keys()) == {0, 1}
    assert obs[0].shape[-1] == 32
    assert obs[1].shape[-1] == 32
