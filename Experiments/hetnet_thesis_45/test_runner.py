"""Launcher tests use synthetic outputs; they never train agents."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_suite as runner


def completed_output(output, opt):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps({'total_steps': opt['nepisodes'] * opt['nsteps']}))
    with (output / 'episode_metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['episode', 'steps', 'avg_reward', 'qos_rate', 'return_val',
                         'capacity_mean', 'learning_steps', 'loss_avg', 'mean_q_avg',
                         'max_q_avg', 'td_error_avg', 'grad_norm_avg', 'success_rate'])
        for i in range(opt['nepisodes']):
            writer.writerow([i, opt['nsteps'], 1, 1, 4, 1, 1, 0.1, 1, 2, 0.1, 0.1, 1])


class RunnerTests(unittest.TestCase):
    def test_exact_matrix_and_ablation(self):
        runs = runner.validate_manifest(runner.SUITE)
        options = {r['id']: runner.read_json(runner.SUITE / r['options']) for r in runs}
        self.assertTrue(all(o['action_strategy'] == 'langevin' for o in options.values()))
        self.assertTrue(all(o['early_termination'] for o in options.values()))
        self.assertTrue(all(o['learning_rate'] == 0.001 for o in options.values()))
        self.assertTrue(all(o['epsilon_policy'] == 'linear_increasing' and o['eps_increment'] == 0.002 for o in options.values()))
        base = options['medium_gcn_ch30_seed42']
        ablated = options['medium_gcn_no_interference_ch30_seed42']
        changed = {k for k in base if not k.startswith('_') and base[k] != ablated[k]}
        self.assertEqual(changed, {'gnn_include_interference_edges'})
        for seed in [42, 123, 456]:
            baseline = options[f'medium_flat_ch30_seed{seed}']
            for repeat in [2, 3]:
                variant = options[f'medium_flat_langevin_repeat{repeat}_ch30_seed{seed}']
                self.assertEqual({k for k in baseline if not k.startswith('_')
                                  and baseline[k] != variant[k]}, set())

    def test_completion_and_numerical_validation(self):
        opt = dict(nepisodes=2, nsteps=4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertTrue(runner.validate_result(output, opt))
            completed_output(output, opt)
            self.assertEqual(runner.validate_result(output, opt), [])
            metric = output / 'episode_metrics.csv'
            metric.write_text(metric.read_text().replace('0.1', 'nan', 1))
            self.assertTrue(any('non-finite' in s for s in runner.validate_result(output, opt)))
            completed_output(output, opt)
            (output / 'summary.json').write_text('{"total_steps": 4}')
            self.assertTrue(any('budget' in s for s in runner.validate_result(output, opt)))

    def test_early_termination_and_incomplete_episodes(self):
        opt = dict(nepisodes=2, nsteps=4, early_termination=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            completed_output(output, dict(nepisodes=2, nsteps=1))
            self.assertEqual(runner.validate_result(output, opt), [])
            self.assertTrue(runner.validate_result(output, dict(opt, early_termination=False)))
            # Shortening individual episodes must not excuse missing episodes.
            completed_output(output, dict(nepisodes=1, nsteps=1))
            self.assertTrue(any('budget' in issue for issue in runner.validate_result(output, opt)))

    def test_short_episode_requires_success(self):
        opt = dict(nepisodes=2, nsteps=4, early_termination=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            completed_output(output, dict(nepisodes=2, nsteps=1))
            path = output / 'episode_metrics.csv'
            with path.open() as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            for row in rows:
                row['success_rate'] = '0'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            self.assertTrue(any('full-QoS' in issue for issue in runner.validate_result(output, opt)))

    def test_sequential_failure_continuation_and_resume(self):
        calls = []
        active = 0
        class FakeProcess:
            def __init__(self, command, **kwargs):
                nonlocal active
                if active:
                    raise AssertionError('Concurrent training processes')
                active += 1
                calls.append(command)
                self.command = command
            def wait(self, timeout=None):
                nonlocal active
                active -= 1
                # First run fails; the remaining 44 must still be attempted.
                if len(calls) == 1:
                    return 1
                opt = runner.read_json(Path(self.command[self.command.index('-c2') + 1]))
                prefix = Path(self.command[self.command.index('--run-name') + 1])
                completed_output(prefix.with_name(prefix.name + '_synthetic'), opt)
                return 0
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / 'results'
            with patch.object(runner, 'RESULTS', results), \
                 patch.object(runner, 'preflight', return_value={'test': True}), \
                 patch.object(runner.subprocess, 'Popen', FakeProcess), \
                 patch.object(runner.subprocess, 'run', return_value=type('Git', (), {'stdout': '', 'stderr': ''})()), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch('sys.argv', ['run_suite.py']):
                    self.assertEqual(runner.main(), 1)
                self.assertEqual(len(calls), 45)
                batch = next(results.glob('batch_*'))
                state = runner.read_json(batch / 'status.json')
                self.assertEqual(sum(v['status'] == 'passed' for v in state.values()), 44)
                with patch('sys.argv', ['run_suite.py', '--resume', str(batch)]):
                    self.assertEqual(runner.main(), 0)
                self.assertEqual(len(calls), 46, 'Resume should retry only the failed run')
                state = runner.read_json(batch / 'status.json')
                self.assertTrue(all(v['status'] == 'passed' for v in state.values()))
                with patch('sys.argv', ['run_suite.py', '--resume', str(batch)]):
                    self.assertEqual(runner.main(), 0)
                self.assertEqual(len(calls), 46, 'Completed batch must not be rerun')


if __name__ == '__main__':
    unittest.main()
