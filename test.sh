#!/usr/bin/env bash
set -e

echo "=== Running NeuralShape Verification Test Suite ==="
if [ -d ".venv" ]; then
    .venv/bin/pytest tests/test_end_to_end.py -v "$@"
else
    pytest tests/test_end_to_end.py -v "$@"
fi
