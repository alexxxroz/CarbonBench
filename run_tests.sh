#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing carbonbench with test dependencies..."
pip install -e ".[dev]" --quiet

echo "Running tests..."
pytest tests/ -v
