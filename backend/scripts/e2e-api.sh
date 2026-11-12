#!/usr/bin/env bash
# Isolated API for Playwright: fresh data dir and database, real worker subprocesses and real
# MLX training/inference on the pinned base model (Metal on Apple Silicon, CPU on Linux CI).
set -euo pipefail
cd "$(dirname "$0")/.."
export TUNELAB_DATA_DIR="${TUNELAB_E2E_DATA_DIR:-var-e2e}"
rm -rf "$TUNELAB_DATA_DIR"
export TUNELAB_CORS_ORIGINS='["http://127.0.0.1:3194"]'
exec uv run uvicorn tunelab.main:app --host 127.0.0.1 --port "${TUNELAB_E2E_API_PORT:-8194}" --log-level warning
