# UARA-DRL

## How to run

### 1) Install dependencies
Minimal requirements are listed in `requirements.txt`.

```
pip install -r requirements.txt
```

### 2) Run a training trial

```
python main.py -c1 Config/config_test_sce.json -c2 Config/config_test_opt.json -n 1
```

Arguments:
- `-c1/--config_path1`: Scenario (sce) JSON file.
- `-c2/--config_path2`: Options (opt) JSON file.
- `-n/--ntrials`: Number of independent trials to run (default 1).

Outputs are written under `Result/run/` (overwritten each run). You will find:
- `opt.json`, `sce.json`: snapshots of the provided configs
- `environment.json`: Python/OS/CUDA summary
- `episode_metrics.csv`: one line per episode
- `step_metrics.csv`: per-step metrics (throttled by `step_log_throttle`)
- `episode_times.csv`: episode durations
- `training_progress.png`: live plot rendered during training
- `checkpoints/`: model checkpoints (if enabled)
- `summary.json`: final summary metrics

Tip: change `plot_x_axis` between `steps` and `episodes` to control the live plot’s x-axis.

## Configuration files

This repository ships with four example configs in `Config/`:

1) `config_1.json` (Scenario example, “sce”)
	 - Defines the telecom environment topology and physical parameters.
	 - Keys include:
		 - `nMBS`, `nPBS`, `nFBS`: number of macro/pico/femto base stations
		 - `rMBS`, `rPBS`, `rFBS`: coverage radii
		 - `BW`: system bandwidth (Hz)
		 - `nChannel`: number of channels
		 - `N0`: noise spectral density (dBm/Hz)
		 - `profit`, `power_cost`, `action_cost`, `negative_cost`: reward shaping terms
		 - `QoS_thr`: QoS threshold (dB)

2) `config_2.json` (Training options, “opt”)
	 - Governs RL agent behavior, learning, logging, and plotting.
	 - Notable keys (also documented in the `_doc` block within the file):
		 - `nagents`: number of agents (UEs)
		 - `capacity`: replay buffer capacity per agent
		 - `learningrate`, `momentum`: RMSprop optimizer settings
		 - `eps_min`, `eps_max`, `eps_increment`: epsilon-greedy schedule (linear growth up to `eps_max`)
		 - `batch_size`, `gamma`, `nepisodes`, `nsteps`, `nupdate`
		 - `enable_plot`, `plot_interval`, `plot_smooth_window`, `plot_x_axis`
		 - `step_log_throttle`: write step metrics every N global steps
		 - `checkpoint_interval`: checkpoint period (0 = disabled)

3) `config_test_sce.json` (Small scenario for quick tests)
	 - Minimal single-MBS setup with fewer channels and a lower QoS threshold—fast to run.

4) `config_test_opt.json` (Lightweight training options)
	 - Small number of agents/steps/episodes suitable for quick validation and CI.
	 - Example settings: `nagents=5`, `nepisodes=5`, `nsteps=50`, plotting per step.

### Choosing which files to pass to `main.py`

- For a quick smoke/validation run:
	- `-c1 Config/config_test_sce.json -c2 Config/config_test_opt.json`
- For a larger experiment:
	- `-c1 Config/config_1.json -c2 Config/config_2.json`


