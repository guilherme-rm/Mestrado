#!/usr/bin/env python3
"""Batch experiment runner for GNN vs Flat comparison experiments.

Usage:
    python scripts/run_experiments.py experiments.yaml
    python scripts/run_experiments.py experiments.yaml --group small
    python scripts/run_experiments.py experiments.yaml --experiments small_gnn medium_gnn
    python scripts/run_experiments.py experiments.yaml --dry-run
    python scripts/run_experiments.py experiments.yaml --parallel 2
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    sce: str
    opt: str
    gnn_enabled: bool = True
    gnn_mode: str = "replace"
    mobility: bool = False
    seeds: List[int] = field(default_factory=lambda: [42])
    deferred_plots: bool = True
    no_resource_plots: bool = False
    description: str = ""
    
    def to_commands(self) -> List[List[str]]:
        """Generate command-line arguments for each seed."""
        commands = []
        for seed in self.seeds:
            cmd = [
                sys.executable, "main.py",
                "-c1", self.sce,
                "-c2", self.opt,
                "--seed", str(seed),
                "--run-name", f"{self.name}_seed{seed}",
            ]
            
            if self.gnn_enabled:
                cmd.append("--gnn-enabled")
                cmd.extend(["--gnn-mode", self.gnn_mode])
            else:
                cmd.append("--no-gnn")
            
            if self.mobility:
                cmd.append("--mobility")
            else:
                cmd.append("--no-mobility")
            
            if self.deferred_plots:
                cmd.append("--deferred-plots")
            
            if self.no_resource_plots:
                cmd.append("--no-resource-plots")
            
            commands.append(cmd)
        
        return commands


def load_experiments(yaml_path: str) -> Dict[str, Any]:
    """Load experiment configurations from YAML file."""
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def parse_experiment(exp_dict: Dict, defaults: Dict) -> ExperimentConfig:
    """Parse a single experiment configuration with defaults."""
    # Merge defaults with experiment-specific settings
    config = {**defaults, **exp_dict}
    
    return ExperimentConfig(
        name=config["name"],
        sce=config["sce"],
        opt=config["opt"],
        gnn_enabled=config.get("gnn_enabled", True),
        gnn_mode=config.get("gnn_mode", "replace"),
        mobility=config.get("mobility", False),
        seeds=config.get("seeds", [42]),
        deferred_plots=config.get("deferred_plots", True),
        no_resource_plots=config.get("no_resource_plots", False),
        description=config.get("description", ""),
    )


def run_single_experiment(cmd: List[str], experiment_name: str, seed: int, 
                          dry_run: bool = False, verbose: bool = True) -> Dict:
    """Run a single experiment and return results."""
    start_time = time.time()
    run_id = f"{experiment_name}_seed{seed}"
    
    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return {
            "run_id": run_id,
            "status": "dry_run",
            "duration": 0,
            "command": " ".join(cmd),
        }
    
    print(f"\n{'='*60}")
    print(f"Starting: {run_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=False,
        )
        
        duration = time.time() - start_time
        status = "success" if result.returncode == 0 else "failed"
        
        return {
            "run_id": run_id,
            "status": status,
            "returncode": result.returncode,
            "duration": duration,
            "command": " ".join(cmd),
            "stdout": result.stdout if not verbose else None,
            "stderr": result.stderr if not verbose else None,
        }
    
    except Exception as e:
        return {
            "run_id": run_id,
            "status": "error",
            "error": str(e),
            "duration": time.time() - start_time,
            "command": " ".join(cmd),
        }


def run_experiments_sequential(experiments: List[ExperimentConfig], 
                                dry_run: bool = False,
                                verbose: bool = True) -> List[Dict]:
    """Run all experiments sequentially."""
    all_results = []
    total_runs = sum(len(exp.seeds) for exp in experiments)
    current_run = 0
    
    for exp in experiments:
        print(f"\n{'#'*60}")
        print(f"# Experiment: {exp.name}")
        print(f"# Description: {exp.description}")
        print(f"# Seeds: {exp.seeds}")
        print(f"{'#'*60}")
        
        for cmd in exp.to_commands():
            current_run += 1
            seed = int(cmd[cmd.index("--seed") + 1])
            print(f"\n[{current_run}/{total_runs}] Running {exp.name} with seed {seed}")
            
            result = run_single_experiment(cmd, exp.name, seed, dry_run, verbose)
            all_results.append(result)
            
            if result["status"] == "success":
                print(f"✓ Completed in {result['duration']:.1f}s")
            elif result["status"] != "dry_run":
                print(f"✗ Failed: {result.get('error', 'see stderr')}")
    
    return all_results


def run_experiments_parallel(experiments: List[ExperimentConfig],
                              max_workers: int = 2,
                              dry_run: bool = False) -> List[Dict]:
    """Run experiments in parallel (one experiment at a time, but multiple seeds)."""
    all_results = []
    
    # Flatten all commands with metadata
    all_runs = []
    for exp in experiments:
        for cmd in exp.to_commands():
            seed = int(cmd[cmd.index("--seed") + 1])
            all_runs.append((cmd, exp.name, seed))
    
    total_runs = len(all_runs)
    print(f"\nRunning {total_runs} experiments with {max_workers} parallel workers")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_experiment, cmd, name, seed, dry_run, False): 
            (name, seed) for cmd, name, seed in all_runs
        }
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            name, seed = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                status_icon = "✓" if result["status"] == "success" else "✗"
                print(f"[{completed}/{total_runs}] {status_icon} {name}_seed{seed}: "
                      f"{result['status']} ({result['duration']:.1f}s)")
            except Exception as e:
                print(f"[{completed}/{total_runs}] ✗ {name}_seed{seed}: error - {e}")
                all_results.append({
                    "run_id": f"{name}_seed{seed}",
                    "status": "error",
                    "error": str(e),
                })
    
    return all_results


def save_batch_results(results: List[Dict], output_dir: str = "Result"):
    """Save batch run results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = Path(output_dir) / f"batch_results_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "total_runs": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "total_duration": sum(r.get("duration", 0) for r in results),
        "results": results,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBatch results saved to: {output_path}")
    return output_path


def print_summary(results: List[Dict]):
    """Print a summary of all experiment runs."""
    print(f"\n{'='*60}")
    print("BATCH EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    errors = [r for r in results if r["status"] == "error"]
    
    print(f"Total runs: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Errors: {len(errors)}")
    
    total_time = sum(r.get("duration", 0) for r in results)
    print(f"Total time: {total_time/60:.1f} minutes")
    
    if failed:
        print(f"\nFailed runs:")
        for r in failed:
            print(f"  - {r['run_id']}")
    
    if errors:
        print(f"\nError runs:")
        for r in errors:
            print(f"  - {r['run_id']}: {r.get('error', 'unknown')}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run batch experiments from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python scripts/run_experiments.py experiments.yaml
  
  # Run only small scenario experiments
  python scripts/run_experiments.py experiments.yaml --group small
  
  # Run specific experiments
  python scripts/run_experiments.py experiments.yaml --experiments small_gnn medium_gnn
  
  # Dry run (show commands without executing)
  python scripts/run_experiments.py experiments.yaml --dry-run
  
  # Run with 2 parallel workers
  python scripts/run_experiments.py experiments.yaml --parallel 2
        """,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiments YAML configuration file",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Run only experiments in this group (defined in YAML)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Run only these specific experiments by name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Number of parallel workers (default: sequential)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress subprocess output (only show progress)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save batch results to JSON",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        return 1
    
    config = load_experiments(args.config)
    defaults = config.get("defaults", {})
    groups = config.get("groups", {})
    
    # Parse all experiments
    all_experiments = [
        parse_experiment(exp, defaults) 
        for exp in config.get("experiments", [])
    ]
    
    # Filter experiments based on arguments
    if args.experiments:
        experiments = [e for e in all_experiments if e.name in args.experiments]
        if not experiments:
            print(f"Error: No experiments found with names: {args.experiments}")
            print(f"Available: {[e.name for e in all_experiments]}")
            return 1
    elif args.group:
        if args.group not in groups:
            print(f"Error: Group '{args.group}' not found")
            print(f"Available groups: {list(groups.keys())}")
            return 1
        group_names = groups[args.group]
        experiments = [e for e in all_experiments if e.name in group_names]
    else:
        experiments = all_experiments
    
    if not experiments:
        print("No experiments to run")
        return 0
    
    print(f"Experiments to run: {[e.name for e in experiments]}")
    total_runs = sum(len(e.seeds) for e in experiments)
    print(f"Total runs (with seeds): {total_runs}")
    
    # Run experiments
    start_time = time.time()
    
    if args.parallel and args.parallel > 1:
        results = run_experiments_parallel(
            experiments, 
            max_workers=args.parallel,
            dry_run=args.dry_run,
        )
    else:
        results = run_experiments_sequential(
            experiments,
            dry_run=args.dry_run,
            verbose=not args.quiet,
        )
    
    total_duration = time.time() - start_time
    
    # Print summary
    print_summary(results)
    print(f"Total batch time: {total_duration/60:.1f} minutes")
    
    # Save results
    if not args.no_save and not args.dry_run:
        save_batch_results(results)
    
    # Return non-zero if any failures
    failed_count = sum(1 for r in results if r["status"] in ("failed", "error"))
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
