import torch
from dotdic import DotDic
from telecom.scenario import Scenario
from telecom.base_station import BS


def make_sce():
    return DotDic(
        {
            "nMBS": 1,
            "nPBS": 1,
            "nFBS": 1,
            "nChannel": 2,
            "rMBS": 500,
            "rPBS": 250,
            "rFBS": 50,
        }
    )


def test_scenario_counts():
    sce = make_sce()
    scenario = Scenario(sce)
    assert scenario.BS_Number() == sce.nMBS + sce.nPBS + sce.nFBS
    loc_mbs, loc_pbs, loc_fbs = scenario.BS_Location()
    assert loc_mbs.shape == (sce.nMBS, 2)
    assert loc_pbs.shape == (sce.nPBS, 2)
    assert loc_fbs.shape == (sce.nFBS, 2)


def test_bs_receive_power_inside_radius():
    sce = make_sce()
    bs = BS(sce, 0, "MBS", [0, 0], sce.rMBS)
    p = bs.receive_power(100)
    assert p > 0


def test_bs_receive_power_outside_radius():
    sce = make_sce()
    bs = BS(sce, 0, "MBS", [0, 0], 10.0)
    p = bs.receive_power(50.0)
    assert p == 0.0
