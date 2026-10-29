#!/usr/bin/env bash
set -e

PORT=${1:-8000}
HOST=${2:-0.0.0.0}

echo "=== Starting NeuralShape Server on http://${HOST}:${PORT} ==="
if [ -d ".venv" ]; then
    .venv/bin/uvicorn backend.app.main:app --host "$HOST" --port "$PORT" --reload
else
    uvicorn backend.app.main:app --host "$HOST" --port "$PORT" --reload
fi
