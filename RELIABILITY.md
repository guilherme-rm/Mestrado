# Reliability changes and experiment compatibility

The pre-review implementation is preserved in commit `25729bc`. This branch fixes
configuration and execution defects; it does not introduce the planned token-based
Transformer or a new experiment batch.

## Configuration and reproducibility

`functions/configuration.py` resolves defaults and validates experiments before
resource allocation. Missing `DotDic` attributes now raise `AttributeError`, so
`getattr(..., default)` and `hasattr` follow normal Python behavior. An explicitly
configured zero reward coefficient remains zero. Runtime switches belong to the
run rather than mutating global constants. Unsupported graph augmentation and
heterogeneous encoding fail clearly; the old parallel-optimization flag emits a
warning and records its actual sequential behavior.

Use `device: "cpu"`, `"cuda:0"`, or `"auto"`. Explicit unavailable CUDA requests
fail; legacy `cuda: -1` selects CPU and legacy nonnegative indices remain portable
to a CPU-only host. `auto` retains automatic multi-GPU selection.

The CLI seed is recorded in `opt.json`. Trial i uses `(base_seed + i) % 2**32`.
HetNet layout and mobility use separate derived seeds, recorded as `layout_seed`
and `mobility_seed`; either can be explicitly configured for a single run.
Subsequent multi-trial runs derive fresh streams from their trial seed.
Metadata records `reproducibility_version: 2` and the Git revision/dirty state.
The effective batch size is saved after automatic device adjustment.

**Existing seeded layouts will not be reproduced by this revision.** Geometry
getters no longer resample femtocells or consume random numbers, and layout RNGs
are separated from learning. Existing result folders are untouched; use the
snapshot commit to reproduce the previous implementation. Hardware determinism
is not guaranteed by seeds alone.

## Learning semantics

The following defaults preserve the established learning choices:

- `bootstrap_on_termination: true`: legacy bootstrapping remains enabled. Set it
  to false for episodic targets that stop bootstrapping after full-QoS termination.
  Replay stores true termination separately; horizon truncation still bootstraps.
- `replay_sampling: "legacy"`: tensor replay samples with replacement and deque
  replay without replacement. Explicit `with_replacement` and `without_replacement`
  policies allow controlled comparisons across buffer implementations.
- `target_update_clock: "episode"`: retains the episode-local update clock.
  `"global"` uses the training interaction count. Target synchronization runs on
  its own clock even when the optimizer cadence skips an interaction.
- `learn_every: 1`: every interaction is stored; only optimizer updates are
  throttled when this setting is increased. Shared networks update after all
  agents have inserted their transitions.
- `gnn_encoder_training: "fixed"`: graph encoders remain random fixed feature
  extractors, with existing dropout during training and evaluation mode during
  inference. End-to-end encoder training is deliberately unsupported here.

Langevin exploration, the linear-increasing schedule, and configured learning
rates are unchanged. Graphs with no previous allocation no longer invent a last-
channel assignment from the -1 sentinel; this can affect initial graph embeddings.
Non-finite action values or loss now stop training rather than hiding a failure.

## Code boundaries

- `telecom/radio.py`: shared path-loss/power formulas and units.
- `telecom/hetnet_reward.py`: HetNet allocation evaluation.
- `rl/interaction.py`: observation construction, geometry-only movement, frozen
  greedy selection, and joint reward/QoS evaluation after action selection.
- `rl/optimization.py`: Double-DQN updates and diagnostics.
- `rl/networks.py`: explicit action-value model factory (currently MLP only).
- `functions/training_components.py`: schedules and metric definitions.
- `functions/main_classes.py`: experiment setup and training orchestration.

Legacy imports remain supported. CLI evaluation and the maintained mobility
notebook reuse the interaction helpers. A movement call does not evaluate QoS;
observations combine current geometry with feedback from the last allocation.

Cell-free rewards now use one network snapshot/activity sample per joint step,
and UE indexing remains correct when two users occupy the same position. A
one-AP serving cluster contains exactly one AP. These are behavior corrections,
not claims of equivalence with previous cell-free results.

## Outputs, checkpoints, and interruption

Run directories use unique suffixes and exclusive creation. Loggers are flushed
on normal exit, Python exceptions, Ctrl-C, and CLI SIGTERM; SIGKILL and power loss
cannot execute cleanup. `run_status.json` reports completion, failure, interruption,
and final plotting errors. A plotting failure does not discard completed weights
or numerical summaries.

Checkpoint format 2 records architecture, observation schema/dimension, and source
revision. Format 1 checkpoints remain loadable. Snapshots remain for inference,
not exact training resume. The maintained `Notebooks/checkpoint_mobility.ipynb` is
tracked without generated outputs; its checkpoint path is intentionally editable.

## Validation

Run the complete suite with:

```bash
~/venvs/PyTorch/bin/python -m pytest -q
```

Tests cover configuration, output collisions, seeds, CPU and available CUDA/AMP
training, all current observation paths, checkpoint round trips, frozen evaluation,
movement ordering, replay/target semantics, shared updates, cell-free snapshots,
and interrupted/failed run cleanup. Smoke tests use temporary folders, small
networks, and short horizons: they verify execution, not long-run convergence.
GitHub Actions runs CPU tests on every pull request. `requirements-lock.txt` pins
the tested direct dependencies; select the appropriate PyTorch wheel for the host.

For reproducible timing analysis, pass the desired run explicitly:

```bash
python analyze_metrics.py /path/to/run/time_metrics.txt
```
