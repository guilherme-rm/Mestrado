"""Entry point for running wireless resource allocation training using refactored classes."""

import argparse
import copy
import json

import torch

from dotdic import DotDic
from functions.main_classes import ExperimentManager


def run_multiple_trials(opt, sce, ntrials: int):
    """Run multiple independent trials."""
    all_metrics = []
    
    for i in range(ntrials):
        print(f"\n{'='*60}")
        print(f"Starting Trial {i+1}/{ntrials}")
        print(f"{'='*60}\n")
        
        # Deep copy to ensure independent trials
        trial_opt = copy.deepcopy(opt)
        trial_sce = copy.deepcopy(sce)
        
        # Run trial using ExperimentManager
        experiment = ExperimentManager(trial_opt, trial_sce)
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
    args = parser.parse_args()
    
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
    print(f"-----------------------------\n")
    
    # Run training
    all_metrics = run_multiple_trials(opt, sce, args.ntrials)
    
    print(f"\n{'='*60}")
    print(f"All {args.ntrials} trial(s) completed successfully!")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())