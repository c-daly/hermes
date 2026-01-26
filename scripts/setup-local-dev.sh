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

# Verify
poetry run python -c "from logos_config.ports import get_repo_ports; print(f'Hermes ports: {get_repo_ports(\"hermes\")}')"
echo "Setup complete. Run 'poetry run pytest tests/unit -v' to verify."
