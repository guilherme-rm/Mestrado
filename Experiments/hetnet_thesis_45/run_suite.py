#!/usr/bin/env python3
"""Sequential, restartable launcher; does not modify the training implementation."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

SUITE = Path(__file__).resolve().parent
REPO = SUITE.parent.parent
RESULTS = REPO / 'Result' / 'hetnet_thesis_45'


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_manifest(folder):
    runs = read_json(folder / 'manifest.json')['runs']
    assert len(runs) == 45, 'Expected 45 runs'
    assert len({r['id'] for r in runs}) == 45, 'Duplicate run IDs'
    assert len({r['configuration'] for r in runs}) == 15
    assert Counter(r['study'] for r in runs) == {
        'scale_observation': 18, 'dnn_size': 6, 'channels': 12,
        'interference_edges': 3, 'baseline_repeats': 6}
    signatures = {}
    signatures_by_id = {}
    for r in runs:
        assert set(r['id']) <= set('abcdefghijklmnopqrstuvwxyz0123456789_')
        sce, opt = read_json(folder / r['scenario']), read_json(folder / r['options'])
        assert opt['seed'] == r['seed'] and r['seed'] in [42, 123, 456]
        assert opt['environment_type'] == 'hetnet' and not opt['mobility_enabled']
        assert (opt['nepisodes'], opt['nsteps']) == (500, 100)
        assert opt['early_termination'] and not opt['use_amp']
        assert not opt['batch_size_auto'] and not opt['multi_gpu_auto']
        assert opt['gnn_observation_mode'] == 'replace'
        assert not opt['gnn_transformer_enabled'] and not opt['gnn_use_attention']
        assert opt['gnn_conv_type'] == 'gcn'
        assert opt['action_strategy'] == 'langevin', 'This suite requires Langevin in every run'
        assert opt['learning_rate'] == 0.001
        assert opt['epsilon_policy'] == 'linear_increasing' and opt['eps_increment'] == 0.002
        assert sce['nPBS'] <= 4 * sce['nMBS'], 'Unsupported pico layout'
        signature = json.dumps([{k:v for k,v in d.items() if not k.startswith('_')}
                                for d in [sce, opt]], sort_keys=True)
        if r['study'] == 'baseline_repeats':
            expected = f'medium_flat_ch30_seed{r["seed"]}'
            assert r.get('duplicate_of') == expected
            assert signatures_by_id.get(expected) == signature, 'Repeat must match its baseline'
        else:
            assert 'duplicate_of' not in r
            assert signature not in signatures, 'Undeclared duplicate experiment'
            signatures[signature] = r['id']
        signatures_by_id[r['id']] = signature
    assert len(signatures) == 39, 'Expected 39 distinct seeded configurations and 6 repeats'
    for name in {r['configuration'] for r in runs}:
        assert sorted(r['seed'] for r in runs if r['configuration'] == name) == [42, 123, 456]
    return runs


def source_fingerprint():
    paths = list(REPO.glob('*.py'))
    for directory in ['rl', 'telecom', 'telecom_cellfree', 'functions']:
        paths.extend((REPO / directory).rglob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def preflight():
    code = '''import json, torch, torch_geometric, numpy, matplotlib, psutil, pandas
import constants
assert not constants.SHARED_AGENT_NETWORKS, 'Suite requires independent networks'
print(json.dumps(dict(torch=torch.__version__, pyg=torch_geometric.__version__,
 cuda=torch.cuda.is_available(), device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')))
'''
    checked = subprocess.run([sys.executable, '-c', code], cwd=REPO,
                             capture_output=True, text=True)
    if checked.returncode:
        raise RuntimeError('Dependency/configuration check failed:\n' + checked.stderr)
    help_check = subprocess.run([sys.executable, str(REPO / 'main.py'), '--help'],
                                cwd=REPO, capture_output=True, text=True)
    if help_check.returncode:
        raise RuntimeError('Training entry point cannot load:\n' + help_check.stderr)
    print('Environment:', checked.stdout.strip(), flush=True)
    return json.loads(checked.stdout.strip())


def command_for(run, folder, prefix):
    return [sys.executable, '-u', str(REPO / 'main.py'),
            '-c1', str(folder / run['scenario']), '-c2', str(folder / run['options']),
            '--seed', str(run['seed']), '--run-name', str(prefix), '-n', '1']


def validate_result(output, opt):
    """Reject incomplete or non-finite logs, including apparently successful exits."""
    issues = []
    if not output or not (output / 'summary.json').is_file():
        return ['Missing final summary']
    summary = read_json(output / 'summary.json')
    metric_file = output / 'episode_metrics.csv'
    if not metric_file.is_file():
        return ['Missing episode metrics']
    with metric_file.open() as stream:
        rows = list(csv.DictReader(stream))
    actual_steps = sum(int(row['steps']) for row in rows)
    if len(rows) != opt['nepisodes'] or summary.get('total_steps') != actual_steps:
        issues.append('Incomplete episode/step budget')
    required = ['avg_reward', 'qos_rate', 'return_val', 'capacity_mean']
    learning = ['loss_avg', 'mean_q_avg', 'max_q_avg', 'td_error_avg', 'grad_norm_avg']
    bad = Counter()
    learned = 0
    for index, row in enumerate(rows):
        steps = int(row['steps'])
        valid_steps = 1 <= steps <= opt['nsteps'] if opt.get('early_termination', False) else steps == opt['nsteps']
        if int(row['episode']) != index or not valid_steps:
            issues.append('Unexpected episode sequence or step count')
            break
        if steps < opt['nsteps'] and not (0 < float(row.get('success_rate') or 0) <= 1):
            issues.append('Short episode without recorded full-QoS success')
        learned += int(row.get('learning_steps') or 0)
        for key in required + learning:
            value = row.get(key, '')
            if value == '' and key not in required:
                continue
            try:
                if not math.isfinite(float(value)):
                    bad[key] += 1
            except (TypeError, ValueError):
                bad[key] += 1
    if not learned:
        issues.append('No learning updates recorded')
    if bad:
        issues.append('Invalid/non-finite episode metrics: ' + str(dict(bad)))
    return issues


def save_status(batch, status):
    write_json(batch / 'status.json', status)
    with (batch / 'status.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'status', 'exit_code', 'seconds', 'output', 'issues'])
        for name, item in status.items():
            writer.writerow([name, item['status'], item.get('exit_code', ''),
                             item.get('seconds', ''), item.get('output', ''),
                             '; '.join(item.get('issues', []))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Print all 45 commands; no training')
    mode.add_argument('--check', action='store_true', help='Validate manifest and environment only')
    mode.add_argument('--smoke-test', action='store_true', help='Test each of 15 configurations with 4 UEs, 2 episodes x 4 steps')
    mode.add_argument('--resume', type=Path, help='Resume a batch directory, retrying incomplete/invalid runs from scratch')
    parser.add_argument('--timeout-hours', type=float, default=0,
                        help='Optional limit per run; zero means no limit')
    args = parser.parse_args()
    if args.timeout_hours < 0:
        parser.error('--timeout-hours cannot be negative')
    runs = validate_manifest(SUITE)
    print('Validated: 45 Langevin runs (39 distinct seeded configurations + 6 baseline repeats), seeds 42/123/456.', flush=True)
    if args.dry_run:
        for r in runs:
            print(shlex.join(command_for(r, SUITE, f'hetnet_thesis_45/DRY_RUN/{r["id"]}')))
        return 0
    environment = preflight()
    if args.check:
        return 0
    RESULTS.mkdir(parents=True, exist_ok=True)
    # Kernel releases this lock even if the launcher is interrupted.
    with (RESULTS / '.runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another suite launcher is already running')
        fingerprint = source_fingerprint()
        if args.resume:
            batch = args.resume.expanduser().resolve()
            if batch.parent != RESULTS.resolve():
                raise RuntimeError('Resume directory must be a batch directly under ' + str(RESULTS))
            metadata = read_json(batch / 'batch.json')
            if metadata['source_sha256'] != fingerprint:
                raise RuntimeError('Training source changed since this batch; start a new batch instead')
            if metadata['environment'] != environment or metadata['python'] != sys.executable:
                raise RuntimeError('Python environment changed; start a new batch instead')
            if metadata['smoke_test']:
                raise RuntimeError('Smoke batches are not resumable; run --smoke-test again')
            folder = batch / 'configs'
            runs = validate_manifest(folder)
        else:
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            batch = RESULTS / (('smoke_' if args.smoke_test else 'batch_') + stamp)
            batch.mkdir()
            folder = batch / 'configs'
            folder.mkdir()
            shutil.copy2(SUITE / 'manifest.json', folder / 'manifest.json')
            for name in ['scenarios', 'options']:
                shutil.copytree(SUITE / name, folder / name)
            for name in ['run_all.sh', 'run_suite.py', 'README.md']:
                shutil.copy2(SUITE / name, batch / name)
            write_json(batch / 'batch.json', dict(source_sha256=fingerprint,
                       python=sys.executable, environment=environment, smoke_test=args.smoke_test,
                       created=datetime.now(timezone.utc).isoformat()))
            git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True)
            (batch / 'git_commit.txt').write_text(git.stdout or git.stderr)
            diff = subprocess.run(['git', 'diff', '--', '*.py'], cwd=REPO, capture_output=True, text=True)
            (batch / 'working_tree_python.diff').write_text(diff.stdout)
        if args.smoke_test:
            runs = [r for r in runs if r['seed'] == 42]
            for r in runs:
                opt = read_json(folder / r['options'])
                opt.update(nagents=4, nepisodes=2, nsteps=4, batch_size=2,
                           capacity=16, min_memory_for_learning=2,
                           enable_plot=False, network_plot_enabled=False, resource_plot_enabled=False)
                write_json(folder / r['options'], opt)
        status = read_json(batch / 'status.json') if (batch / 'status.json').exists() else {}
        (batch / 'logs').mkdir(exist_ok=True)
        print('Batch:', batch, flush=True)
        for index, r in enumerate(runs, 1):
            previous = status.get(r['id'], {})
            opt = read_json(folder / r['options'])
            if previous.get('status') == 'passed':
                output = Path(previous['output'])
                if not validate_result(output, opt):
                    print(f'[{index}/{len(runs)}] SKIP {r["id"]} (validated)', flush=True)
                    continue
            attempt = int(previous.get('attempt', 0)) + 1
            prefix = batch / f'{r["id"]}_attempt{attempt}'
            cmd = command_for(r, folder, prefix)
            log_path = batch / 'logs' / f'{r["id"]}_attempt{attempt}.log'
            item = dict(status='running', attempt=attempt, command=cmd, log=str(log_path))
            status[r['id']] = item
            save_status(batch, status)
            print(f'[{index}/{len(runs)}] RUN {r["id"]}\n  Log: {log_path}', flush=True)
            started = time.monotonic()
            interrupted = False
            issues = []
            with log_path.open('w') as log:
                env = dict(os.environ, MPLBACKEND='Agg', PYTHONUNBUFFERED='1', PYTHONHASHSEED=str(r['seed']))
                process = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, start_new_session=True)
                try:
                    code = process.wait(timeout=args.timeout_hours * 3600 or None)
                except (KeyboardInterrupt, subprocess.TimeoutExpired) as error:
                    interrupted = isinstance(error, KeyboardInterrupt)
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = 130 if interrupted else 124
                    issues.append('Interrupted' if interrupted else 'Per-run timeout')
            outputs = sorted(batch.glob(prefix.name + '_*'))
            output = outputs[-1] if outputs else None
            if code:
                issues.append(f'Training process exited with code {code}')
            else:
                try:
                    issues.extend(validate_result(output, opt))
                except Exception as error:
                    issues.append('Result validation failed: ' + str(error))
            item.update(status='interrupted' if interrupted else ('failed' if issues else 'passed'),
                        exit_code=code, seconds=round(time.monotonic()-started, 2),
                        output=str(output) if output else '', issues=issues)
            save_status(batch, status)
            print(f'  {item["status"].upper()}: ' + ('; '.join(issues) if issues else 'complete, finite logged metrics'), flush=True)
            if interrupted:
                print('Resume with: bash ' + shlex.quote(str(SUITE / 'run_all.sh')) + ' --resume ' + shlex.quote(str(batch)))
                return 130
        failed = sum(status[r['id']]['status'] != 'passed' for r in runs)
        print(f'Finished: {len(runs)-failed} passed; {failed} failed. Status: {batch / "status.csv"}', flush=True)
        return 1 if failed else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, AssertionError) as error:
        print('ERROR:', error, file=sys.stderr)
        raise SystemExit(2)
