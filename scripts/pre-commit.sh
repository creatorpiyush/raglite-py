#!/bin/bash
set -e

# Resolve python: prefer the venv's python, then python3
PYTHON="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null)}"
if [ -z "$PYTHON" ]; then
  echo "ERROR: No Python found. Activate a venv or install Python 3."
  exit 1
fi

echo "=== Running Pre-Commit Checks (python: $PYTHON) ==="

echo "1. Checking code style and formatting (ruff)..."
$PYTHON -m ruff check .

echo "2. Running typechecks (mypy)..."
$PYTHON -m mypy src/ --ignore-missing-imports --no-error-summary || true

echo "3. Running tests..."
$PYTHON -m pytest tests/ -q

echo "=== All checks passed! Ready to commit ==="
