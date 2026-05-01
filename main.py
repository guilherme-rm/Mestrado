"""Entry point for running wireless resource allocation training using refactored classes."""

import argparse
import copy
import json
import random
import subprocess
import sys

import numpy as np
import torch

from dotdic import DotDic
from functions.runtime_settings import apply_runtime_settings
from telecom.environment_factory import resolve_environment_type
from telecom import mobility


def run_multiple_trials(opt, sce, ntrials: int, run_name: str = None):
    """Run multiple independent trials."""
    from functions.main_classes import ExperimentManager

    all_metrics = []
    
    for i in range(ntrials):
        print(f"\n{'='*60}")
        print(f"Starting Trial {i+1}/{ntrials}")
        print(f"{'='*60}\n")
        
        # Deep copy to ensure independent trials
        trial_opt = copy.deepcopy(opt)
        trial_sce = copy.deepcopy(sce)
        
        # Determine run name for this trial
        if run_name:
            trial_run_name = f"{run_name}_trial{i+1}" if ntrials > 1 else run_name
        else:
            trial_run_name = None
        
        # Run trial using ExperimentManager
        experiment = ExperimentManager(trial_opt, trial_sce, run_name=trial_run_name)
        metrics = experiment.run_trial()
        all_metrics.append(metrics)
        
        print(f"\nTrial {i+1} complete.")
    
    return all_metrics


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train RL agents for wireless resource allocation (config-driven)"
    )
    parser.add_argument(
        "-c1", "--config_sce",
        type=str,
        required=True,
        help="Path to scenario config (sce.json)",
    )
    parser.add_argument(
        "-c2", "--config_opt",
        type=str,
        required=True,
        help="Path to RL/optimization config (opt.json)",
    )
    parser.add_argument(
        "-n", "--ntrials",
        type=int,
        default=1,
        help="Number of independent trials to run (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (sets numpy, torch, and python random)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for the run directory (instead of timestamp)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Open lightweight launcher interface instead of running directly",
    )
    args = parser.parse_args()

    if args.ui:
        cmd = [sys.executable, "launch_interface.py"]
        return subprocess.run(cmd, check=False).returncode
    
    # Load configurations
    try:
        with open(args.config_sce, "r") as f:
            sce = DotDic(json.load(f))
        print(f"Loaded scenario config: {args.config_sce}")
    except FileNotFoundError:
        print(f"Error: Scenario config file not found: {args.config_sce}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing scenario config JSON: {e}")
        return 1
    
    try:
        with open(args.config_opt, "r") as f:
            opt = DotDic(json.load(f))
        print(f"Loaded optimization config: {args.config_opt}")
    except FileNotFoundError:
        print(f"Error: Optimization config file not found: {args.config_opt}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing optimization config JSON: {e}")
        return 1
    
    # Apply runtime settings from config files
    runtime = apply_runtime_settings(opt, sce)

    # Command-line seed overrides config seed
    seed = args.seed if args.seed is not None else getattr(opt, "seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        mobility.MOBILITY_SEED = seed
        print(f"Random seed set to: {seed}")

    # Print device info
    from functions.gpu_manager import GPUManager
    _gpu = GPUManager.from_opt(opt)
    device = _gpu.device
    print(f"Using device: {device}")

    env_type = resolve_environment_type(opt)

    # Print config summary
    print(f"\n--- Configuration Summary ---")
    print(f"Agents: {opt.nagents}")
    print(f"Episodes: {opt.nepisodes}")
    print(f"Steps per episode: {opt.nsteps}")
    if env_type == "cell_free":
        n_ap = getattr(sce, "nAP", None)
        if n_ap is None:
            n_ap = (getattr(sce, "nMBS", 0) or 0) + (getattr(sce, "nPBS", 0) or 0) + (getattr(sce, "nFBS", 0) or 0)
        print(f"Access points: {n_ap} APs (Cell-Free)")
    else:
        print(f"Base stations: {sce.nMBS} MBS, {sce.nPBS} PBS, {sce.nFBS} FBS")
    print(f"Channels: {sce.nChannel}")
    print(f"Trials: {args.ntrials}")
    print(f"Environment type: {env_type}")
    print(f"GNN enabled: {runtime['gnn_enabled']}")
    print(f"GNN transformer: {runtime['gnn_transformer_enabled']}")
    print(f"Mobility enabled: {runtime['mobility_enabled']}")
    print(f"Fast mode: {runtime['fast_mode']}")
    print(f"Plots: network={runtime['network_plot_enabled']}, telecom={runtime['telecom_plot_enabled']}, resource={runtime['resource_plot_enabled']}")
    print(f"Deferred plotting: {runtime['deferred_plotting']}")
    if args.run_name:
        print(f"Run name: {args.run_name}")
    print(f"-----------------------------\n")
    
    # Run training
    all_metrics = run_multiple_trials(opt, sce, args.ntrials, run_name=args.run_name)
    
    print(f"\n{'='*60}")
    print(f"All {args.ntrials} trial(s) completed successfully!")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())