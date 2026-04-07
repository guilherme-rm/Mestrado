#!/usr/bin/env python3
"""Aggregate and compare results from multiple experiment runs.

Usage:
    python scripts/aggregate_results.py Result/
    python scripts/aggregate_results.py Result/ --filter "gnn"
    python scripts/aggregate_results.py Result/ --compare small_gnn small_flat
    python scripts/aggregate_results.py Result/ --output comparison_results.csv
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    run_name: str
    run_path: Path
    summary: Dict[str, Any]
    episode_metrics: Optional[pd.DataFrame] = None
    step_metrics: Optional[pd.DataFrame] = None
    
    @property
    def experiment_name(self) -> str:
        """Extract experiment name (without seed suffix)."""
        # Pattern: experiment_name_seedN_timestamp or experiment_name_seedN
        match = re.match(r"(.+)_seed\d+", self.run_name)
        if match:
            return match.group(1)
        return self.run_name
    
    @property
    def seed(self) -> Optional[int]:
        """Extract seed from run name."""
        match = re.search(r"_seed(\d+)", self.run_name)
        if match:
            return int(match.group(1))
        return None
    
    @property
    def gnn_enabled(self) -> bool:
        """Check if GNN was enabled for this run."""
        return self.summary.get("gnn_enabled", False)
    
    @property
    def final_reward(self) -> float:
        """Get final average reward."""
        if self.episode_metrics is not None and "avg_reward" in self.episode_metrics.columns:
            return self.episode_metrics["avg_reward"].iloc[-1]
        return self.summary.get("final_avg_reward", 0.0)
    
    @property
    def best_reward(self) -> float:
        """Get best average reward achieved."""
        if self.episode_metrics is not None and "avg_reward" in self.episode_metrics.columns:
            return self.episode_metrics["avg_reward"].max()
        return self.summary.get("best_avg_reward", 0.0)
    
    @property
    def training_time(self) -> float:
        """Get total training time in seconds."""
        return self.summary.get("training_time_seconds", 0.0)


def load_experiment_result(run_path: Path) -> Optional[ExperimentResult]:
    """Load results from a single run directory."""
    summary_path = run_path / "summary.json"
    if not summary_path.exists():
        # Try experiment_summary.json as fallback
        summary_path = run_path / "experiment_summary.json"
        if not summary_path.exists():
            return None
    
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    
    # Load episode metrics
    episode_metrics = None
    episode_path = run_path / "episode_metrics.csv"
    if episode_path.exists():
        try:
            episode_metrics = pd.read_csv(episode_path)
        except Exception:
            pass
    
    # Load step metrics (optional, can be large)
    step_metrics = None
    step_path = run_path / "step_metrics.csv"
    if step_path.exists():
        try:
            step_metrics = pd.read_csv(step_path)
        except Exception:
            pass
    
    return ExperimentResult(
        run_name=run_path.name,
        run_path=run_path,
        summary=summary,
        episode_metrics=episode_metrics,
        step_metrics=step_metrics,
    )


def discover_runs(result_dir: Path, filter_pattern: Optional[str] = None) -> List[ExperimentResult]:
    """Discover all experiment runs in result directory."""
    results = []
    
    for item in result_dir.iterdir():
        if not item.is_dir():
            continue
        
        # Skip non-run directories
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        
        # Apply filter if provided
        if filter_pattern and not re.search(filter_pattern, item.name, re.IGNORECASE):
            continue
        
        result = load_experiment_result(item)
        if result:
            results.append(result)
    
    return results


def group_by_experiment(results: List[ExperimentResult]) -> Dict[str, List[ExperimentResult]]:
    """Group results by experiment name (across seeds)."""
    groups = defaultdict(list)
    for result in results:
        groups[result.experiment_name].append(result)
    return dict(groups)


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "n": 0}
    
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
    }


def aggregate_experiment_group(results: List[ExperimentResult]) -> Dict[str, Any]:
    """Aggregate statistics for a group of experiment runs (same experiment, different seeds)."""
    if not results:
        return {}
    
    # Get first result for config info
    first = results[0]
    
    # Collect metrics across seeds
    final_rewards = [r.final_reward for r in results]
    best_rewards = [r.best_reward for r in results]
    training_times = [r.training_time for r in results]
    
    return {
        "experiment_name": first.experiment_name,
        "gnn_enabled": first.gnn_enabled,
        "num_seeds": len(results),
        "seeds": [r.seed for r in results],
        "final_reward": compute_statistics(final_rewards),
        "best_reward": compute_statistics(best_rewards),
        "training_time": compute_statistics(training_times),
        "runs": [r.run_name for r in results],
    }


def compare_experiments(groups: Dict[str, Dict], 
                        experiment_names: Optional[List[str]] = None) -> pd.DataFrame:
    """Create comparison table between experiments."""
    if experiment_names:
        groups = {k: v for k, v in groups.items() if k in experiment_names}
    
    rows = []
    for name, stats in sorted(groups.items()):
        row = {
            "Experiment": name,
            "GNN": "Yes" if stats["gnn_enabled"] else "No",
            "Seeds": stats["num_seeds"],
            "Final Reward (mean)": f"{stats['final_reward']['mean']:.4f}",
            "Final Reward (std)": f"{stats['final_reward']['std']:.4f}",
            "Best Reward (mean)": f"{stats['best_reward']['mean']:.4f}",
            "Best Reward (std)": f"{stats['best_reward']['std']:.4f}",
            "Time (min)": f"{stats['training_time']['mean']/60:.1f}",
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def compare_gnn_vs_flat(groups: Dict[str, Dict]) -> pd.DataFrame:
    """Create paired comparison between GNN and flat experiments."""
    # Find pairs (same scenario, different GNN setting)
    pairs = []
    
    for name, stats in groups.items():
        if stats["gnn_enabled"]:
            # Find matching flat experiment
            flat_name = name.replace("_gnn", "_flat")
            if flat_name in groups:
                flat_stats = groups[flat_name]
                
                # Compute improvement
                gnn_reward = stats["final_reward"]["mean"]
                flat_reward = flat_stats["final_reward"]["mean"]
                improvement = ((gnn_reward - flat_reward) / abs(flat_reward) * 100 
                              if flat_reward != 0 else 0)
                
                pairs.append({
                    "Scenario": name.replace("_gnn", ""),
                    "GNN Reward": f"{gnn_reward:.4f} ± {stats['final_reward']['std']:.4f}",
                    "Flat Reward": f"{flat_reward:.4f} ± {flat_stats['final_reward']['std']:.4f}",
                    "Improvement (%)": f"{improvement:+.2f}%",
                    "GNN Time (min)": f"{stats['training_time']['mean']/60:.1f}",
                    "Flat Time (min)": f"{flat_stats['training_time']['mean']/60:.1f}",
                })
    
    return pd.DataFrame(pairs)


def generate_learning_curves(groups: Dict[str, List[ExperimentResult]], 
                              output_path: Path,
                              metric: str = "avg_reward"):
    """Generate learning curve comparison plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.tab10.colors
    
    for i, (name, results) in enumerate(sorted(groups.items())):
        # Aggregate episode metrics across seeds
        all_rewards = []
        max_len = 0
        
        for result in results:
            if result.episode_metrics is not None and metric in result.episode_metrics.columns:
                rewards = result.episode_metrics[metric].values
                all_rewards.append(rewards)
                max_len = max(max_len, len(rewards))
        
        if not all_rewards:
            continue
        
        # Pad shorter arrays with NaN
        padded = np.full((len(all_rewards), max_len), np.nan)
        for j, rewards in enumerate(all_rewards):
            padded[j, :len(rewards)] = rewards
        
        # Compute mean and std
        mean_rewards = np.nanmean(padded, axis=0)
        std_rewards = np.nanstd(padded, axis=0)
        episodes = np.arange(1, max_len + 1)
        
        color = colors[i % len(colors)]
        ax.plot(episodes, mean_rewards, label=name, color=color)
        ax.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                       alpha=0.2, color=color)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Learning Curves Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Learning curves saved to: {output_path}")


def print_summary(groups: Dict[str, Dict]):
    """Print summary to console."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    comparison_df = compare_experiments(groups)
    print("\n" + comparison_df.to_string(index=False))
    
    gnn_vs_flat_df = compare_gnn_vs_flat(groups)
    if not gnn_vs_flat_df.empty:
        print("\n" + "-"*80)
        print("GNN vs FLAT COMPARISON")
        print("-"*80)
        print("\n" + gnn_vs_flat_df.to_string(index=False))
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and compare experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "result_dir",
        type=str,
        help="Directory containing experiment run folders",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Regex pattern to filter experiment names",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to compare",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for comparison table",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Output path for learning curves plot",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Output JSON file for aggregated statistics",
    )
    
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        return 1
    
    # Discover and load results
    print(f"Scanning {result_dir} for experiment results...")
    results = discover_runs(result_dir, args.filter)
    
    if not results:
        print("No experiment results found")
        return 1
    
    print(f"Found {len(results)} experiment runs")
    
    # Group by experiment
    result_groups = group_by_experiment(results)
    print(f"Grouped into {len(result_groups)} experiments")
    
    # Aggregate statistics
    aggregated = {
        name: aggregate_experiment_group(group)
        for name, group in result_groups.items()
    }
    
    # Print summary
    print_summary(aggregated)
    
    # Save outputs
    if args.output:
        df = compare_experiments(aggregated, args.compare)
        df.to_csv(args.output, index=False)
        print(f"Comparison table saved to: {args.output}")
    
    if args.json:
        with open(args.json, "w") as f:
            json.dump(aggregated, f, indent=2)
        print(f"Aggregated statistics saved to: {args.json}")
    
    if args.plot:
        generate_learning_curves(result_groups, Path(args.plot))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
