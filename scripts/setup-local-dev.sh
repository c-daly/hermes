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

# Install with local override
cd "$REPO_ROOT"

# Install deps first (this installs logos-foundry from git)
echo "Installing dependencies..."
poetry install -E dev

# Override with local editable install
echo "Installing local logos-foundry as editable..."
poetry run pip install -e "$LOGOS_PATH"

echo ""
echo "SUCCESS: Local development environment ready"
echo "logos-foundry is installed as editable from: $LOGOS_PATH"
echo ""
echo "Note: The pip editable install overrides the git-installed version."
echo "To verify: poetry run python -c 'import logos_config; print(logos_config.__file__)'"
