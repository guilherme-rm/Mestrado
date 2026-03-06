"""Tests for telecom network topology plotter."""

import torch
from dotdic import DotDic
from telecom.scenario import Scenario
from rl.training import create_agents
from functions.telecom_network_plot import TelecomNetworkPlotter


def make_configs():
    opt = DotDic(
        {
            "nagents": 5,
            "capacity": 20,
            "learning_rate": 1e-3,
            "momentum": 0.0,
            "batch_size": 4,
            "gamma": 0.9,
            "nupdate": 2,
            "action_selection_strategy": "epsilon_greedy",
        }
    )
    sce = DotDic(
        {
            "nMBS": 1,
            "nPBS": 2,
            "nFBS": 2,
            "nChannel": 3,
            "rMBS": 500,
            "rPBS": 250,
            "rFBS": 50,
            "N0": -100,
            "BW": 1e6,
            "QoS_thr": 0,
            "negative_cost": -0.5,
        }
    )
    return opt, sce


def test_telecom_plotter_creates_file(tmp_path):
    """Test that the plotter creates an output file."""
    opt, sce = make_configs()
    device = torch.device("cpu")
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, device)
    
    out_file = tmp_path / "network_topology.png"
    plotter = TelecomNetworkPlotter(
        scenario=scenario,
        agents=agents,
        enabled=True,
        plot_interval=1,
        out_path=str(out_file),
    )
    
    # Simulate some actions (agent 0 -> BS 0, channel 0; agent 1 -> BS 1, channel 1, etc.)
    actions = [i % (scenario.BS_Number() * sce.nChannel) for i in range(opt.nagents)]
    
    plotter.update(step=0, episode=0, actions=actions)
    
    assert out_file.exists()
    plotter.close()


def test_telecom_plotter_disabled_no_file(tmp_path):
    """Test that disabled plotter does not create a file."""
    opt, sce = make_configs()
    device = torch.device("cpu")
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, device)
    
    out_file = tmp_path / "network_topology.png"
    plotter = TelecomNetworkPlotter(
        scenario=scenario,
        agents=agents,
        enabled=False,
        plot_interval=1,
        out_path=str(out_file),
    )
    
    plotter.update(step=0, episode=0, actions=[0] * opt.nagents)
    
    assert not out_file.exists()
    plotter.close()


def test_telecom_plotter_throttling(tmp_path):
    """Test that plotter respects plot_interval throttling."""
    opt, sce = make_configs()
    device = torch.device("cpu")
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, device)
    
    out_file = tmp_path / "network_topology.png"
    plotter = TelecomNetworkPlotter(
        scenario=scenario,
        agents=agents,
        enabled=True,
        plot_interval=5,  # Only render every 5 steps
        out_path=str(out_file),
    )
    
    # Step 1-4 should not create file (throttled)
    for step in range(1, 5):
        plotter.update(step=step, episode=0, actions=None)
    
    # File should not exist yet (step 0 wasn't called, steps 1-4 throttled)
    assert not out_file.exists()
    
    # Step 5 should trigger render
    plotter.update(step=5, episode=0, actions=None)
    assert out_file.exists()
    
    plotter.close()


def test_telecom_plotter_no_actions(tmp_path):
    """Test that plotter works without action connections."""
    opt, sce = make_configs()
    device = torch.device("cpu")
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, device)
    
    out_file = tmp_path / "network_topology.png"
    plotter = TelecomNetworkPlotter(
        scenario=scenario,
        agents=agents,
        enabled=True,
        plot_interval=1,
        out_path=str(out_file),
        show_connections=True,
    )
    
    # Update without actions - should still render
    plotter.update(step=0, episode=0, actions=None)
    
    assert out_file.exists()
    plotter.close()
