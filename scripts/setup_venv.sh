#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   PYTHON_BIN=python3.10 VENV_DIR=.venv REQ_FILE=setup/requirements.txt bash scripts/setup_venv.sh
# Defaults:
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="${REQ_FILE:-setup/requirements.txt}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python binary not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install --no-deps -r "$REQ_FILE"

echo "Virtualenv created at $VENV_DIR. Activate with: source $VENV_DIR/bin/activate"
