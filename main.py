"""Entry point for running wireless resource allocation training using refactored classes."""

import argparse
import copy
import json
import random

import numpy as np
import torch

from dotdic import DotDic
from functions.main_classes import ExperimentManager
import constants
from telecom import mobility


def run_multiple_trials(opt, sce, ntrials: int, run_name: str = None):
    """Run multiple independent trials."""
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
        description="Train RL agents for wireless resource allocation (refactored version)"
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
        "--deferred-plots",
        action="store_true",
        help="Render plots only at the end of training instead of during (reduces I/O overhead)",
    )
    parser.add_argument(
        "--no-resource-plots",
        action="store_true",
        help="Disable CPU/GPU resource usage plotting",
    )
    parser.add_argument(
        "--gnn-enabled",
        action="store_true",
        default=None,
        help="Enable GNN-based observation encoding",
    )
    parser.add_argument(
        "--no-gnn",
        action="store_true",
        help="Disable GNN-based observation encoding (use flat observations)",
    )
    parser.add_argument(
        "--gnn-mode",
        type=str,
        choices=["replace", "augment"],
        default=None,
        help="GNN observation mode: 'replace' (GNN only) or 'augment' (GNN + flat obs)",
    )
    parser.add_argument(
        "--mobility",
        action="store_true",
        default=None,
        help="Enable UE mobility simulation",
    )
    parser.add_argument(
        "--no-mobility",
        action="store_true",
        help="Disable UE mobility simulation",
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
        "--slow-mode",
        action="store_true",
        help="Disable fast mode: enables plots, full logging, all features",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true", 
        help="Disable all plotting (telecom, network metrics, resource metrics)",
    )
    args = parser.parse_args()
    
    # Slow mode: enable expensive features for full diagnostics
    if args.slow_mode:
        constants.FAST_MODE = False
        constants.NETWORK_PLOT_ENABLED = True
        constants.TELECOM_PLOT_ENABLED = True
        constants.RESOURCE_PLOT_ENABLED = True
        constants.DEFAULT_STEP_LOG_THROTTLE = 10
        constants.DEFERRED_PLOTTING = False
        print("Slow mode enabled: full plots and logging")
    
    # Disable all plots
    if args.no_plots:
        constants.NETWORK_PLOT_ENABLED = False
        constants.TELECOM_PLOT_ENABLED = False
        constants.RESOURCE_PLOT_ENABLED = False
    
    # Apply command-line flags to constants
    if args.deferred_plots:
        constants.DEFERRED_PLOTTING = True
    if args.no_resource_plots:
        constants.RESOURCE_PLOT_ENABLED = False
    
    # GNN configuration
    if args.no_gnn:
        constants.GNN_ENABLED = False
    elif args.gnn_enabled:
        constants.GNN_ENABLED = True
    if args.gnn_mode:
        constants.GNN_OBSERVATION_MODE = args.gnn_mode
    
    # Mobility configuration
    if args.no_mobility:
        mobility.MOBILITY_ENABLED = False
    elif args.mobility:
        mobility.MOBILITY_ENABLED = True
    
    # Random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # Also set the mobility seed
        mobility.MOBILITY_SEED = args.seed
        print(f"Random seed set to: {args.seed}")
    
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
    
    # Print device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Print config summary
    print(f"\n--- Configuration Summary ---")
    print(f"Agents: {opt.nagents}")
    print(f"Episodes: {opt.nepisodes}")
    print(f"Steps per episode: {opt.nsteps}")
    print(f"Base stations: {sce.nMBS} MBS, {sce.nPBS} PBS, {sce.nFBS} FBS")
    print(f"Channels: {sce.nChannel}")
    print(f"Trials: {args.ntrials}")
    print(f"GNN enabled: {constants.GNN_ENABLED}")
    print(f"Mobility enabled: {mobility.MOBILITY_ENABLED}")
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