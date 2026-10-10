# HetNet policy checkpoints and evaluation

HetNet training now saves `checkpoints/final.pt` at the end of training. The existing
`checkpoint_interval` configuration controls additional snapshots after every N
completed episodes. Set it to 0 to disable periodic saving; final saving remains
on unless `save_final_checkpoint` is false or `--no-final-checkpoint` is supplied.
Each snapshot is written to a temporary file and atomically replaces its destination.

## Train and save

From the repository root, with the existing virtual environment:

```bash
source ~/venvs/PyTorch/bin/activate
python main.py \
  -c1 Config/hetnet/config_sce_small.json \
  -c2 Config/opt/config_opt_small.json \
  --seed 42 --checkpoint-interval 100 --run-name small_checkpoint
```

This uses the supplied configurations unchanged except for the explicit checkpoint
interval override. Existing `-c3` architecture configurations remain supported.
The console prints the exact checkpoint paths, for example:

```text
Result/small_checkpoint_<timestamp>/checkpoints/episode_00100.pt
Result/small_checkpoint_<timestamp>/checkpoints/final.pt
```

## Run a trained policy without training

Replace `<timestamp>` with the actual training directory:

```bash
python main.py \
  --evaluate 'Result/small_checkpoint_<timestamp>/checkpoints/final.pt' \
  --eval-episodes 1 --eval-steps 100
```

Evaluation reads its configurations directly from the checkpoint, so do not pass
`-c1`, `-c2`, or `-c3`. CPU is the default; `--eval-device cuda` selects the GPU.
The saved maximum episode length is used when `--eval-steps` is omitted. The saved
early-termination setting is honored. All actions are greedy argmax decisions;
exploration and dropout are disabled, gradients are disabled, and no transitions
are stored or optimization/target updates performed. In particular, evaluation
does not reuse the training Langevin exploration rule.

Results are written under `Result/eval_<timestamp>/`. `--run-name PATH` chooses an
explicit evaluation output directory, which must not already exist. Outputs:

- `actions.csv`: the joint allocation at each step (agent order matches the checkpoint).
- `evaluation_metrics.csv`: per-episode mean reward per agent per step, mean QoS
  satisfaction, final QoS satisfaction, whether all users satisfied QoS simultaneously,
  executed steps, and elapsed time.
- `evaluation.json`: source checkpoint, seed, horizon, and evaluation settings.

These are inference results, so they should be reported separately from exploratory
training metrics. Repeating episodes in this static deterministic setting can
produce identical trajectories; it does not provide independent scenario samples.

## What is preserved and current limits

Each checkpoint contains all agents' policy and target weights, graph encoder
weights when enabled, effective architecture/configuration settings, UE and base
station positions, and the completed episode index (zero-based) and interaction count.
The observation starts from the same zero-QoS initialization used at training episode
boundaries. Evaluation is a new episode, not continuation of an interrupted step.
The model definitions and radio-model code must remain compatible with the checkpoint.

This is a **policy checkpoint**, not an exact training-resume snapshot: replay buffers,
optimizer state, and random-generator state are not saved. It cannot yet resume
training through a CLI command. It supports HetNet; other environments retain their
training behavior without policy checkpointing.

Evaluation currently holds the saved UE positions fixed, even if training used mobility.
An evaluation mobility loop can later move UEs, refresh radio/graph state, and select
new allocations with the same frozen policies. That option is not implemented here.

Previously completed runs that did not save model weights cannot be loaded using
this feature; their CSV results are not sufficient to reconstruct a trained policy.
