#!/bin/bash
set -e
echo "=== Hermes Local Development Setup ==="

# Install dependencies (pulls logos-foundry from git tag)
poetry install --extras "dev ml"

# Copy .env.example if .env doesn't exist
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "Created .env from .env.example"
fi

# Download V-JEPA weights (optional — skip if JEPA provider not needed)
if [ "${DOWNLOAD_JEPA_WEIGHTS:-0}" = "1" ]; then
    WEIGHTS_DIR="${JEPA_WEIGHTS_DIR:-$HOME/.cache/hermes/jepa}"
    mkdir -p "$WEIGHTS_DIR"
    WEIGHTS_PATH="$WEIGHTS_DIR/vitH14.pth"
    if [ ! -f "$WEIGHTS_PATH" ]; then
        echo "Downloading V-JEPA ViT-H/14 weights to $WEIGHTS_PATH ..."
        # Weights are hosted on the Meta V-JEPA HuggingFace repo.
        # Set JEPA_WEIGHTS_URL to override the download source.
        WEIGHTS_URL="${JEPA_WEIGHTS_URL:-https://dl.fbaipublicfiles.com/jepa/vitH14.pth}"
        curl -fL "$WEIGHTS_URL" -o "$WEIGHTS_PATH"
        echo "V-JEPA weights downloaded."
    else
        echo "V-JEPA weights already present at $WEIGHTS_PATH"
    fi
    export JEPA_WEIGHTS_PATH="$WEIGHTS_PATH"
else
    echo "Skipping V-JEPA weight download (set DOWNLOAD_JEPA_WEIGHTS=1 to enable)"
fi

# Verify
poetry run python -c "from logos_config.ports import get_repo_ports; print(f'Hermes ports: {get_repo_ports(\"hermes\")}')"
echo "Setup complete. Run 'poetry run pytest tests/unit -v' to verify."
