#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

poetry install --extras "dev ml" 2>/dev/null || poetry install --extras dev
echo "Dependencies installed. Starting Hermes dev server..."
poetry run uvicorn hermes.main:app --reload --host 0.0.0.0 --port 17000
