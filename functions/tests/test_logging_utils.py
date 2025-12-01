import json
from functions import (
    RunDirectoryManager,
    save_config,
    save_environment_snapshot,
    EpisodeMetricsLogger,
    StepMetricsLogger,
    write_summary,
)
from dotdic import DotDic


def test_run_directory_and_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = RunDirectoryManager(root="Result", prefix="testrun")
    opt = DotDic({"alpha": 1, "beta": 2})
    sce = DotDic({"gamma": 3})
    save_config(run_dir, opt, sce)
    save_environment_snapshot(run_dir)
    assert (run_dir.path / "opt.json").exists()
    assert (run_dir.path / "sce.json").exists()
    assert (run_dir.path / "environment.json").exists()


def test_episode_and_step_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = RunDirectoryManager(root="Result", prefix="metrics")
    ep_logger = EpisodeMetricsLogger(run_dir)
    step_logger = StepMetricsLogger(run_dir, throttle=2)
    for ep in range(2):
        ep_logger.log(
            episode=ep,
            steps=10,
            avg_reward=0.5,
            qos_rate=0.8,
            capacity_mean=5.0,
            epsilon_last=0.3,
            duration_seconds=1.2,
        )
    for step in range(5):
        step_logger.log(
            global_step=step,
            episode=0,
            step_in_episode=step,
            epsilon=0.1,
            mean_reward=0.2,
            qos_mean=0.9,
            capacity_sum_mbps=3.3,
        )
    ep_logger.close()
    step_logger.close()
    ep_lines = (run_dir.path / "episode_metrics.csv").read_text().strip().splitlines()
    step_lines = (run_dir.path / "step_metrics.csv").read_text().strip().splitlines()
    assert len(ep_lines) == 1 + 2
    assert len(step_lines) == 1 + 3  # throttle -> steps 0,2,4


def test_summary_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = RunDirectoryManager(root="Result", prefix="summary")
    write_summary(run_dir, {"a": 1, "b": 2})
    data = json.loads((run_dir.path / "summary.json").read_text())
    assert data["a"] == 1 and data["b"] == 2
