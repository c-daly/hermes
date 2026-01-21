# Dependency Troubleshooting

## Common Issues

### "Package not found" for logos-foundry

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement logos-foundry
```

**Solutions:**

1. **Use local development mode** (recommended for active development):
   ```bash
   ./scripts/setup-local-dev.sh
   ```

2. **Clear Poetry cache:**
   ```bash
   poetry cache clear pypi --all
   poetry cache clear _default_cache --all
   rm -rf ~/.cache/pypoetry/cache
   poetry lock --no-update
   poetry install
   ```

3. **Check network/auth:**
   ```bash
   # Test git access
   git ls-remote https://github.com/c-daly/logos.git
   ```

### Poetry vs Pip Conflicts

**Symptoms:**
- `ImportError` for packages that are installed
- Different versions reported by `pip list` vs `poetry show`

**Root Cause:** Mixed pip and poetry installs in same environment.

**Solution:** Use Poetry exclusively:

```bash
# Remove conda/pip installed packages
pip uninstall logos-foundry logos-config -y 2>/dev/null || true

# Reinstall with Poetry
poetry install -E dev
```

### Conda Interference

**Symptoms:**
- Poetry creates venv but Python uses conda packages
- `which python` shows conda path

**Solution:** Deactivate conda or use Poetry's bundled env:

```bash
# Option 1: Deactivate conda
conda deactivate

# Option 2: Force Poetry to use its own Python
poetry env remove --all
poetry env use python3.11
poetry install -E dev

# Option 3: Run commands through poetry run
poetry run pytest tests/
```

### Version Conflicts Between Repos

**Symptoms:**
```
Because hermes depends on logos-foundry (v0.1.0) and apollo depends on logos-foundry (main)
```

**Solution:** Align versions across repos:

1. Check what version each repo expects:
   ```bash
   grep logos-foundry ../hermes/pyproject.toml
   grep logos-foundry ../apollo/pyproject.toml
   ```

2. Update all repos to same tag (e.g., `@v0.1.0`)

3. If developing across repos, use local paths:
   ```bash
   # In each repo's pyproject.local.toml:
   logos-foundry = { path = "../logos", develop = true }
   ```

## CI-Specific Issues

### GitHub Actions Cache Stale

**Symptoms:** CI fails but local works, especially after logos update.

**Solution:** Bust the cache:

```yaml
# In .github/workflows/ci.yml, update cache key:
key: deps-${{ hashFiles('**/poetry.lock') }}-v2  # increment version
```

### Docker Build Fails on Git Deps

**Symptoms:**
```
ERROR: Cannot find git repository at https://github.com/c-daly/logos.git
```

**Solution:** Use build args or multi-stage build:

```dockerfile
# Option 1: Copy local logos into build context
COPY --from=logos-builder /app/logos /logos
RUN pip install /logos

# Option 2: Use GitHub token for private repos
ARG GITHUB_TOKEN
RUN pip install git+https://${GITHUB_TOKEN}@github.com/c-daly/logos.git@v0.1.0
```

## Diagnostic Commands

```bash
# Show installed packages and sources
poetry show --tree

# Show what version of logos_config is actually loaded
poetry run python -c "import logos_config; print(logos_config.__file__)"

# Verify no pip/conda contamination
poetry run pip list | grep logos

# Check Poetry environment
poetry env info
```
