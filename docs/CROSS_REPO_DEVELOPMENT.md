# Cross-Repository Development

When working on changes that span multiple LOGOS repos (logos, hermes, apollo, sophia), follow this workflow.

## Setup

### Clone All Repos Side-by-Side

```bash
mkdir -p ~/projects/logos-dev
cd ~/projects/logos-dev

git clone https://github.com/c-daly/logos.git
git clone https://github.com/c-daly/hermes.git
git clone https://github.com/c-daly/apollo.git
git clone https://github.com/c-daly/sophia.git
```

### Configure Local Dependencies

In each dependent repo, run:

```bash
cd hermes && ./scripts/setup-local-dev.sh
cd ../apollo && ./scripts/setup-local-dev.sh
cd ../sophia && ./scripts/setup-local-dev.sh
```

This configures each repo to use your local logos clone.

## Making Cross-Repo Changes

### Example: Adding New Config to logos-foundry

1. **Make change in logos:**
   ```bash
   cd logos
   git checkout -b feature/new-config
   # Edit logos_config/...
   poetry install
   poetry run pytest tests/
   ```

2. **Test in hermes (using local logos):**
   ```bash
   cd ../hermes
   # Already using local logos via pyproject.local.toml
   poetry run pytest tests/
   ```

3. **Test in apollo:**
   ```bash
   cd ../apollo
   poetry run pytest tests/
   ```

4. **When ready to merge:**
   - PR logos first, get it merged
   - Create new tag: `git tag v0.2.0 && git push --tags`
   - PR hermes with updated `@v0.2.0` in pyproject.toml
   - PR apollo with updated `@v0.2.0` in pyproject.toml

### Example: Updating Hermes API

1. **Update OpenAPI contract in logos:**
   ```bash
   cd logos
   # Edit contracts/hermes.openapi.yaml
   # Regenerate SDK: make generate-hermes-sdk
   ```

2. **Implement in hermes:**
   ```bash
   cd ../hermes
   # Implement new endpoint
   poetry run pytest tests/
   ```

3. **Update apollo to use new endpoint:**
   ```bash
   cd ../apollo
   # Update code to call new hermes endpoint
   poetry run pytest tests/
   ```

4. **Merge order:**
   logos (contract + SDK) → hermes (implementation) → apollo (consumer)

## Environment Isolation

### Per-Repo Virtual Environments

Each repo has its own Poetry-managed venv:

```bash
# Hermes venv
poetry env info  # in hermes/

# Apollo venv
poetry env info  # in apollo/
```

### Avoid Conda Conflicts

If using conda, either:
1. Deactivate it: `conda deactivate`
2. Or use Poetry's explicit env: `poetry run <command>`

## Troubleshooting

See `docs/DEPENDENCY_TROUBLESHOOTING.md` for common issues.
