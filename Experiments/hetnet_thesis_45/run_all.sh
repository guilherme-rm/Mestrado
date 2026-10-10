#!/usr/bin/env bash
set -euo pipefail
suite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-$HOME/venvs/PyTorch/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    printf 'Python not found: %s\nSet PYTHON_BIN to the Python executable in your PyTorch environment.\n' "$python_bin" >&2
    exit 2
fi
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
exec "$python_bin" "$suite_dir/run_suite.py" "$@"
