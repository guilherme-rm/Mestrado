from functions.live_plot import RealTimeStepPlotter


def test_plot_file_created(tmp_path):
    out_file = tmp_path / "progress.png"
    plotter = RealTimeStepPlotter(enabled=True, plot_interval=1, out_path=str(out_file))
    for s in range(10):
        plotter.update(
            step=s,
            epsilon=0.5,
            mean_reward=float(s),
            qos=0.8,
            capacity_sum_mbps=10.0 + s,
        )
    assert out_file.exists()


def test_disabled_no_file(tmp_path):
    out_file = tmp_path / "progress.png"
    plotter = RealTimeStepPlotter(
        enabled=False, plot_interval=1, out_path=str(out_file)
    )
    for s in range(5):
        plotter.update(
            step=s, epsilon=0.1, mean_reward=0.0, qos=0.0, capacity_sum_mbps=0.0
        )
    assert not out_file.exists()
