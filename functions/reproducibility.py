"""Explicit run seeds and independent layout/mobility random streams."""
import random
import secrets
import subprocess
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_seeds(opt):
    if getattr(opt, 'seed', None) is None:
        opt.seed = secrets.randbelow(2**32)
    if not isinstance(opt.seed, int) or not 0 <= opt.seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    derived = np.random.SeedSequence(opt.seed).generate_state(2)
    for key, value in zip(('layout_seed', 'mobility_seed'), derived):
        if getattr(opt, key, None) is None:
            setattr(opt, key, int(value))
    opt.reproducibility_version = 2
    seed_everything(opt.seed)


def source_revision():
    try:
        root = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=root, text=True))
        return {'commit': revision, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}
