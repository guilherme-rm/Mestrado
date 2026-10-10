# HetNet thesis experiment suite

45 sequential Langevin runs: 13 distinct configurations × seeds 42, 123, 456, plus 6 explicitly labeled same-seed baseline repeats. No training-source changes are required. Each seed has a complete optimization JSON, including its seed and neural-network settings; scenario JSON files are shared. The runner invokes the existing main.py once per run.

## Run

From the repository root:

```bash
bash Experiments/hetnet_thesis_45/run_all.sh
```

The default interpreter is **~/venvs/PyTorch/bin/python** (case-sensitive). No activation or dependency installation is performed. Override it explicitly with `PYTHON_BIN=/path/to/python` if necessary.

For execution after closing the terminal:

```bash
nohup bash Experiments/hetnet_thesis_45/run_all.sh > hetnet_thesis_45.log 2>&1 &
```

The computer must remain running and awake. Closing the terminal with nohup is supported; shutdown or suspension does not permit training to continue.

## Matrix and reused baselines

| Study | New configurations | Runs |
|---|---:|---:|
| Small/medium/large × flat/GCN, 30 channels | 6 | 18 |
| Medium flat compact/wide DQN, 30 channels | 2 | 6 |
| Medium × flat/GCN × 10/50 channels | 4 | 12 |
| Medium GCN without interference edges, 30 channels | 1 | 3 |
| Medium flat Langevin baseline repeat groups, 30 channels | 0 (2 repeat groups) | 6 |
| Total | 13 distinct + 2 repeat groups | 45 |

The medium flat/GCN 30-channel cases are executed once per seed and reused across studies. All runs now use Langevin. The former exploration comparison is retained as two explicitly labeled repeat groups to keep the requested total of 45 runs. They repeat the medium flat 30-channel baseline with the same seeds and parameters; they are not independent seed samples or different exploration methods. The manifest identifies them with duplicate_of. Order: scale/observation, DQN widths, channel counts, edge ablation, baseline repeats.

- Infrastructure: small = 2/8/16 macro/pico/femto stations and 30 UEs; medium = 4/16/32 and 60 UEs; large = 8/32/64 and 120 UEs.
- Baseline DQN: [128, 64, 64]; compact: [64, 32, 32]; wide: [256, 128, 128].
- GCN: 2 layers, hidden/output width 64, replacement observations, attention and Transformer disabled. Interference edges are enabled except in the ablation.
- Every full run: 500 episodes, each with at most 100 steps (up to 50,000 environment steps). Early termination is enabled when all users satisfy QoS.
- Optimizer: existing RMSprop, learning rate 0.001, momentum 0.005, gamma 0.9, batch 64, replay capacity 10,000 per agent, warmup 512 transitions, target update interval 50.
- Exploration in every run: Langevin with `epsilon_policy="linear_increasing"`, `eps_min=0.0`, `eps_max=0.9`, and `eps_increment=0.002`. The current scheduler resets the increment at each episode and applies its existing episode-dependent cap. Langevin uses a minimum temperature of 0.05. The exponential-decay fields remain in the JSON for compatibility but are inactive under this policy. These are the user-selected baseline settings; do not replace them with another schedule or learning rate without instruction.
- Independent agents follow the current constants setting, verified at preflight. No unsupported JSON switch is used for sharing.
- AMP, mobility, automatic batch sizing and automatic multi-GPU selection are disabled. CUDA is selected by the existing GPU manager when available; CPU fallback is possible and will be printed at preflight.
- Learning metrics are collected every step. Episode CSVs are complete; step CSVs are throttled to every 100 global steps. Plots are deferred, with topology plotting disabled equally across configurations.

## Validation and restart

Start a NEW batch after the configuration updates (Langevin, early termination, linear increasing and learning rate 0.001). Existing batch snapshots are immutable and retain their original settings. The updated validator rejects snapshots that do not match the selected baseline. No existing results have been changed.

```bash
bash Experiments/hetnet_thesis_45/run_all.sh --dry-run
bash Experiments/hetnet_thesis_45/run_all.sh --check
bash Experiments/hetnet_thesis_45/run_all.sh --smoke-test
```

Dry-run validates the matrix and prints 45 commands without importing the training stack. Check also imports dependencies and the main entry point. Smoke-test runs each of the 15 configuration/repeat groups once using **4 UEs, 2 episodes × 4 steps**, and a small replay buffer. Smoke results are isolated and are not thesis experiments. Smoke tests validate execution and updates, not full-scale resource capacity or long-run numerical stability.

Each full invocation creates a new `Result/hetnet_thesis_45/batch_<timestamp>/` containing immutable input copies, the manifest, launcher copies, code fingerprint, Git revision/diff, command records, per-run logs, and `status.json` / `status.csv`. Training outputs are created inside that batch. A kernel lock prevents two launchers from this suite running concurrently.

Runs are executed one after another. A process failure, timeout, missing output, missing episodes, inconsistent step totals, or a shortened episode without recorded full-QoS success, absence of learning updates, or non-finite logged reward/learning metrics marks a run failed. The launcher continues with subsequent runs and exits nonzero after the batch if any failed. Result checking occurs when each run exits; it does not stop numerical divergence early. Passing these checks does not establish convergence or scientific validity.

Resume a full batch using its printed directory:

```bash
bash Experiments/hetnet_thesis_45/run_all.sh --resume "Result/hetnet_thesis_45/batch_<timestamp>"
```

Validated successful runs are skipped. Failed/interrupted runs are restarted from episode zero, preserving earlier attempt logs. Mid-training checkpoint resume is not implemented by the training code. The runner refuses to resume with a changed training-source fingerprint or reported Python/PyTorch/PyG/device environment; start a new batch in that case. The recorded environment is not a complete package or driver lockfile.

Optional per-run timeout (default is unlimited):

```bash
bash Experiments/hetnet_thesis_45/run_all.sh --timeout-hours 12
```

## Interpretation limits

The graph encoder currently **does not receive RL optimizer updates**; these are untrained graph-encoder feature experiments, not end-to-end GNN training. Existing dropout remains active in that encoder. No augment mode, heterogeneous parameterization, graph transfer, foundation model, or learned communication is introduced.

Flat and graph observations contain different information, so their comparison does not isolate graph structure alone. All runs use the current implementation, not assumed reproductions of historical results. Three matched seed values are recorded, but different architectures may consume random numbers differently. Metrics describe training under exploration; this suite does not add a separate evaluation-only loop.

All 45 runs together have a maximum budget of 2,250,000 environment steps. Actual counts will be lower when episodes terminate early; compare normalized reward, episode success, step counts, and wall time together. Wall time depends strongly on network size and hardware. Start with the execution smoke check; full-size memory use and long-run stability must still be assessed from the real runs.
