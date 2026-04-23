import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('Result/run_20260422-204136/time_metrics.txt')

# Total row count
total_rows = len(df)

# Identifying step vs episode_end rows
# Based on the structure, episode_end rows have global_step/episode/step_in_episode but null or special values for others?
# In the head, we see columns. Let's assume episode_end rows have a specific signature.
# Looking at the headers: global_step, episode, step_in_episode, did_learn, ...
# If a row represents an episode end, it might have NaN in step-specific columns or we can identify them if they represent the end.
# Actually, the prompt says "how many step rows vs episode_end rows".
# Let's check for NaN values to see if episode_end rows exist in this file.
step_rows = df[df['step_total_seconds'].notna()]
episode_end_rows = df[df['step_total_seconds'].isna()]

# Task 1: Row count and breakdown
print(f"Total rows: {total_rows}")
print(f"Step rows: {len(step_rows)}")
print(f"Episode end rows: {len(episode_end_rows)}")

# Task 2: Stats of step_total_seconds
stats = step_rows['step_total_seconds'].describe(percentiles=[.5, .95])
print("\nStep Total Seconds Stats:")
print(f"Mean: {stats['mean']:.4f}")
print(f"Median: {stats['50%']:.4f}")
print(f"P95: {stats['95%']:.4f}")
print(f"Max: {stats['max']:.4f}")

# Task 3: Components of step rows
components = [
    'run_step_seconds', 'learn_seconds', 'log_step_seconds', 'advance_seconds', 'epsilon_seconds'
]
subcomponents = [
    'gnn_encode_seconds', 'select_actions_seconds', 'telecom_update_seconds', 'record_step_metrics_seconds'
]

print("\nComponents (Mean and Share % of step_total_seconds):")
total_mean = step_rows['step_total_seconds'].mean()
for comp in components:
    comp_mean = step_rows[comp].mean()
    share = (comp_mean / total_mean) * 100
    print(f"{comp}: {comp_mean:.4f} ({share:.2f}%)")

print("\nSubcomponents (Mean and Share % of step_total_seconds):")
for sub in subcomponents:
    sub_mean = step_rows[sub].mean()
    share = (sub_mean / total_mean) * 100
    print(f"{sub}: {sub_mean:.4f} ({share:.2f}%)")

# Task 4: Top 10 slowest steps
top_10 = step_rows.nlargest(10, 'step_total_seconds')
print("\nTop 10 Slowest Steps:")
print(top_10[['global_step', 'episode', 'step_in_episode', 'step_total_seconds', 'run_step_seconds', 'learn_seconds', 'telecom_update_seconds']])

# Task 5: Is telecom_update_seconds a major bottleneck?
telecom_mean = step_rows['telecom_update_seconds'].mean()
telecom_share = (telecom_mean / total_mean) * 100
print(f"\nTelecom Update Share: {telecom_share:.2f}%")
if telecom_share > 20:
    print("Telecom update IS a major bottleneck.")
else:
    print("Telecom update is NOT a major bottleneck.")
