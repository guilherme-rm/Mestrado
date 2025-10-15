"""Quick RL learning smoke test.

Runs a very short training using the project's RL stack to demonstrate that
metrics (avg reward, QoS rate) improve over episodes. Uses tiny episode/step
counts so it completes fast on CPU.

Usage:
    python -m scripts.rl_learning_smoke_test
or  python scripts/rl_learning_smoke_test.py
"""
from __future__ import annotations

import json

import torch

from dotdic import DotDic
from telecom.scenario import Scenario
from rl.training import create_agents
from main import run_episodes  # reuse the core training loop


def default_cfg():
    # Small, fast config
    sce = DotDic({
        'nMBS': 1,
        'nPBS': 0,
        'nFBS': 0,
        'rMBS': 200,
        'rPBS': 100,
        'rFBS': 50,
        'BW': 180000,
        'nChannel': 2,
        'N0': -120,
        'QoS_thr': 0,
        'negative_cost': -0.5,
    })
    opt = DotDic({
        'nagents': 3,
        'capacity': 100,
        'learningrate': 0.005,
        'momentum': 0.0,
        'batch_size': 8,
        'gamma': 0.9,
        'nepisodes': 20,   # longer run
        'nsteps': 120,     # longer per-episode steps
        'nupdate': 10,
        # epsilon linear growth
        'eps_min': 0.05,
        'eps_increment': 0.02,
        'eps_max': 0.8,
        # plotting enabled for visual feedback
        'enable_plot': True,
        'plot_interval': 10,
        'plot_smooth_window': 20,
        'plot_x_axis': 'episodes',  # fewer render calls
        # optional logging throttle
        'step_log_throttle': 5,
    })
    return opt, sce


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt, sce = default_cfg()
    scenario = Scenario(sce)
    agents = create_agents(opt, sce, scenario, device)

    # Use a fixed run directory that is overwritten on each run
    import shutil
    from pathlib import Path

    class FixedRunDir:
        def __init__(self, root: str = 'Result/smoke'):
            self.path = Path(root)
            if self.path.exists():
                shutil.rmtree(self.path)
            self.path.mkdir(parents=True, exist_ok=True)

        def subpath(self, name: str) -> Path:
            p = self.path / name
            p.parent.mkdir(parents=True, exist_ok=True)
            return p

    run_dir = FixedRunDir('Result/smoke')

    metrics = run_episodes(opt, sce, agents, scenario, run_dir)

    # Simple improvement heuristic: compare last vs first 2 episodes
    def mean_last(vals, k=2):
        return sum(vals[-k:]) / max(1, min(k, len(vals)))
    def mean_first(vals, k=2):
        return sum(vals[:k]) / max(1, min(k, len(vals)))

    avg_rew_first = mean_first(metrics['avg_reward'])
    avg_rew_last = mean_last(metrics['avg_reward'])
    qos_first = mean_first(metrics['qos_rate'])
    qos_last = mean_last(metrics['qos_rate'])

    print('\n=== Smoke Test Summary ===')
    print(f"AvgReward: first={avg_rew_first:.3f} -> last={avg_rew_last:.3f}")
    print(f"QoS Rate : first={qos_first:.3f} -> last={qos_last:.3f}")
    improved = (avg_rew_last >= avg_rew_first) and (qos_last >= qos_first)
    print(f"Improved: {improved}")

    # Also write a summary JSON into the run directory
    summary = {
        'avg_reward_first': avg_rew_first,
        'avg_reward_last': avg_rew_last,
        'qos_first': qos_first,
        'qos_last': qos_last,
        'improved': improved,
    }
    (run_dir.subpath('smoke_summary.json')).write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
