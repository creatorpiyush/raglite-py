#!/bin/bash
set -e

# Resolve python: prefer the venv's python, then python3
PYTHON="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null)}"
if [ -z "$PYTHON" ]; then
  echo "ERROR: No Python found. Activate a venv or install Python 3."
  exit 1
fi

echo "=== Running Pre-Release Checklist & Build (python: $PYTHON) ==="

echo "1. Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

echo "2. Checking code style and formatting (ruff)..."
$PYTHON -m ruff check .

echo "3. Running typechecks (mypy)..."
$PYTHON -m mypy src/ --ignore-missing-imports --no-error-summary || true

echo "4. Running full test suite..."
$PYTHON -m pytest tests/ -q

echo "5. Building distribution packages..."
$PYTHON -m build

echo "=== Build succeeded! ==="
echo "Checklist before publishing:"
echo " [ ] Incremented version in pyproject.toml?"
echo " [ ] Documented changes in CHANGELOG.md?"
echo " [ ] Verified documentation is up to date?"
echo "Ready to publish: python -m twine upload dist/*"
