# Enriched flat observations for HetNet

Set `observation_mode` to `enriched` in the optimization JSON. Omitting the key
(or setting it to `qos`) preserves the existing QoS-only inputs. The enriched mode
is for HetNet flat policies and cannot be combined with the graph encoder.

Each agent receives its own vector, in this order:

| Feature | Length | Encoding |
| --- | --- | --- |
| Global last QoS | N UEs | Binary indicators, initially zero |
| Distances to all BSs | B BSs | Current distance / coverage radius; not clipped, so values above 1 remain distinguishable |
| Received powers from all BSs | B BSs | The existing simulator path loss and coverage rule; dBm clipped to [-140, 20] and mapped to [0, 1]; zero outside coverage |
| Previous serving BS | B BSs | One-hot; all zeros before the first action |
| Previous channel | K channels | One-hot; all zeros before the first action |
| Previous SINR | 1 | dB clipped to [-20, 30] and mapped to [0, 1]; initially zero |

Total input size is **N + 3B + K + 1**: 119 for 10 UEs, 26 BSs and 30 channels.
SINR is recovered from the previous capacity using the inverse of the simulator's
Shannon formula. It is not an additional pre-action QoS measurement. Candidate
received powers describe individual links; they do not predict post-allocation
interference from other users.

Features are generated for both ends of each replay transition. Current geometry
is refreshed before action selection; previous association, channel and radio
feedback remain from the last completed allocation. Episode reset clears that
feedback while keeping the actual positions. The existing global QoS state still
controls early termination and metrics.

## First controlled training run

`Config/opt/config_opt_small_enriched_10ue_seed42.json` differs from the existing
10-UE configuration **only** by `observation_mode: "enriched"`. Mobility remains
disabled; no action mask was added. Rewards, Langevin exploration, linear-increasing
schedule, learning rate 0.001, hidden widths [128, 64, 64], replay settings,
500 episodes, 100-step limit, early termination, seed 42 and final-checkpoint
saving remain unchanged. Input width and therefore first-layer parameter count
necessarily increase. A new model must be trained; QoS-only weights cannot be
loaded into the larger input layer.

From the repository root:

```bash
source ~/venvs/PyTorch/bin/activate
python -u main.py \
  -c1 Experiments/hetnet_thesis_45/scenarios/small_ch30.json \
  -c2 Config/opt/config_opt_small_enriched_10ue_seed42.json \
  --seed 42 --checkpoint-interval 0 \
  --run-name small_enriched_10ue_checkpoint_seed42
```

The final checkpoint records the observation mode and can be used with `--evaluate`.
The mobility notebook also supports it: change its `CHECKPOINT` path to the new
`final.pt`, restart the kernel and run all cells. Its current selection is left on
the existing 10-UE QoS-only checkpoint until the enriched run is complete.

This first run isolates the observation change. It does not by itself establish
mobility generalization; compare frozen policies on the same movement trajectory.
