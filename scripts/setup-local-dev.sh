#!/usr/bin/env bash
# Setup local development with editable logos-foundry
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Check for logos repo
LOGOS_PATH="${LOGOS_PATH:-$REPO_ROOT/../logos}"

if [[ ! -d "$LOGOS_PATH" ]]; then
    echo "ERROR: logos repo not found at $LOGOS_PATH"
    echo "Either:"
    echo "  1. Clone logos next to this repo: git clone https://github.com/c-daly/logos.git ../logos"
    echo "  2. Set LOGOS_PATH environment variable to your logos clone"
    exit 1
fi

echo "Using logos at: $LOGOS_PATH"

# Create local override if not exists
if [[ ! -f "$REPO_ROOT/pyproject.local.toml" ]]; then
    cat > "$REPO_ROOT/pyproject.local.toml" << EOF
[tool.poetry.dependencies]
logos-foundry = { path = "$LOGOS_PATH", develop = true }
EOF
    echo "Created pyproject.local.toml with path to logos"
fi

# Install with local override
cd "$REPO_ROOT"

# Remove cached logos-foundry
poetry cache clear pypi --all -n 2>/dev/null || true

# Install deps
echo "Installing dependencies..."
poetry install -E dev

echo ""
echo "SUCCESS: Local development environment ready"
echo "logos-foundry is installed from: $LOGOS_PATH"
echo ""
echo "To verify: poetry run python -c 'import logos_config; print(logos_config.__file__)'"
