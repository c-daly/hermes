# Hermes-Apollo Dependency Stabilization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate recurring dependency conflicts between hermes, apollo, and logos-foundry by standardizing on a single package manager, pinning versions, and creating reproducible environments.

**Architecture:** Replace fragile git-branch dependencies with proper versioned releases. Standardize on Poetry across all repos. Create a local development workflow that doesn't require network access to logos repo.

**Tech Stack:** Poetry (primary), pyproject.toml (PEP 621), GitHub Releases, optional pip constraints file for CI

---

## Problem Analysis

### Current State (Brittle)

1. **Hermes** depends on `logos-foundry @ git+https://github.com/c-daly/logos.git@main`
2. **Apollo** depends on:
   - `logos-sophia-sdk @ git+...@main subdirectory=sdk/python/sophia`
   - `logos-hermes-sdk @ git+...@main subdirectory=sdk/python/hermes`
   - `logos-foundry @ git+...@main`
3. **logos** (foundry) publishes multiple packages from one repo with different versioning needs

### Failure Modes

| Issue | Root Cause | Impact |
|-------|------------|--------|
| "Package not found" | Git dep resolution fails (network, auth, branch change) | Blocks all installs |
| Version conflicts | `@main` is moving target; different repos pin different commits | Poetry solver fails |
| Pip vs Poetry | Some envs use pip, others Poetry; git deps resolve differently | Inconsistent environments |
| Conda interference | Conda's env isolation conflicts with Poetry's venv management | Import errors, wrong versions |

---

## Solution Overview

### Phase 1: Local Development Isolation (Quick Win)
Make `poetry install` work reliably by supporting editable local paths.

### Phase 2: Version Pinning via Git Tags
Replace `@main` with tagged releases like `@v0.1.0`.

### Phase 3: Private PyPI / GitHub Packages (Optional)
Publish logos-foundry and SDKs as proper packages for pip-installable deps.

### Phase 4: CI/CD Alignment
Ensure CI uses exact same resolution as local dev.

---

## Task 1: Create Local Dev Override Configuration

**Files:**
- Create: `pyproject.local.toml` (template for local overrides)
- Modify: `pyproject.toml` - add comments about local dev
- Create: `scripts/setup-local-dev.sh`

**Step 1: Create local override template**

Create file `pyproject.local.toml.example`:

```toml
# Copy this to pyproject.local.toml and adjust paths for your system
# This file is gitignored and allows local editable installs

[tool.poetry.dependencies]
# Override logos-foundry to use local path
logos-foundry = { path = "../logos", develop = true }

# For apollo developers who also work on hermes:
# logos-hermes-sdk = { path = "../logos/sdk/python/hermes", develop = true }
```

**Step 2: Run to verify template is valid**

```bash
cat pyproject.local.toml.example
```
Expected: File contents displayed

**Step 3: Add to .gitignore**

Append to `.gitignore`:
```
pyproject.local.toml
```

**Step 4: Run to verify gitignore**

```bash
grep "pyproject.local.toml" .gitignore
```
Expected: Line found

**Step 5: Create setup script**

Create file `scripts/setup-local-dev.sh`:

```bash
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
```

**Step 6: Make script executable and test**

```bash
chmod +x scripts/setup-local-dev.sh
./scripts/setup-local-dev.sh --help 2>&1 || echo "Script created (no --help yet)"
```

**Step 7: Commit**

```bash
git add pyproject.local.toml.example .gitignore scripts/setup-local-dev.sh
git commit -m "feat: add local dev setup for editable logos-foundry

- Add pyproject.local.toml.example template for local path overrides
- Add setup-local-dev.sh script to automate configuration
- Gitignore pyproject.local.toml to keep personal paths out of repo

This enables reliable local development without network dependency on
the logos git repository."
```

---

## Task 2: Pin logos-foundry to Git Tag

**Files:**
- Modify: `pyproject.toml:30`

**Step 1: Check current logos releases**

```bash
gh release list --repo c-daly/logos --limit 5 || echo "No releases yet"
```

**Step 2: If no release exists, create one in logos repo**

This step requires action in the logos repo:

```bash
cd ../logos
git tag -a v0.1.0 -m "Release 0.1.0 - logos-foundry baseline"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --notes "Baseline release for cross-repo dependency pinning"
```

**Step 3: Update pyproject.toml to use tag**

In `hermes/pyproject.toml`, change line 30 from:
```toml
"logos-foundry @ git+https://github.com/c-daly/logos.git@main ; python_version >= \"3.11\" and python_version < \"4.0\"",
```

To:
```toml
"logos-foundry @ git+https://github.com/c-daly/logos.git@v0.1.0 ; python_version >= \"3.11\" and python_version < \"4.0\"",
```

**Step 4: Verify poetry can resolve**

```bash
poetry lock --no-update
```
Expected: Lock file updated successfully

**Step 5: Test install**

```bash
poetry install -E dev
poetry run python -c "import logos_config; print('OK')"
```
Expected: "OK"

**Step 6: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: pin logos-foundry to v0.1.0 tag

Replace @main with @v0.1.0 for reproducible builds. The @main reference
was causing intermittent failures when logos changed in ways that
broke hermes before hermes could adapt."
```

---

## Task 3: Add Dependency Troubleshooting Doc

**Files:**
- Create: `docs/DEPENDENCY_TROUBLESHOOTING.md`

**Step 1: Write troubleshooting guide**

Create `docs/DEPENDENCY_TROUBLESHOOTING.md`:

```markdown
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
```

**Step 2: Verify file renders**

```bash
head -50 docs/DEPENDENCY_TROUBLESHOOTING.md
```

**Step 3: Commit**

```bash
git add docs/DEPENDENCY_TROUBLESHOOTING.md
git commit -m "docs: add dependency troubleshooting guide

Covers common issues:
- Package not found errors
- Poetry vs pip conflicts
- Conda interference
- Version mismatches between repos
- CI cache staleness"
```

---

## Task 4: Create CI Constraints File

**Files:**
- Create: `constraints.txt`
- Modify: `.github/workflows/ci.yml`

**Step 1: Generate constraints from lock file**

```bash
poetry export -f requirements.txt --without-hashes > constraints.txt
```

**Step 2: Verify constraints file**

```bash
head -20 constraints.txt
grep logos constraints.txt || echo "logos-foundry is git dep, won't appear"
```

**Step 3: Update CI to use constraints**

In `.github/workflows/ci.yml`, find the install step and add:

```yaml
- name: Install dependencies
  run: |
    pip install -c constraints.txt -e ".[dev]"
```

This ensures CI uses exact same versions as local dev.

**Step 4: Commit**

```bash
git add constraints.txt .github/workflows/ci.yml
git commit -m "ci: add constraints.txt for reproducible CI builds

Export Poetry lock to constraints.txt so pip-based CI installs
use identical versions."
```

---

## Task 5: Add Pre-commit Hook for Lock Sync

**Files:**
- Create: `.pre-commit-config.yaml` (if not exists)
- Or modify existing

**Step 1: Check if pre-commit config exists**

```bash
cat .pre-commit-config.yaml 2>/dev/null || echo "No pre-commit config yet"
```

**Step 2: Add poetry-lock check**

Create or update `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/python-poetry/poetry
    rev: 2.0.0
    hooks:
      - id: poetry-check
      - id: poetry-lock
        args: ["--check"]
```

**Step 3: Install pre-commit**

```bash
pip install pre-commit
pre-commit install
```

**Step 4: Test hook**

```bash
pre-commit run poetry-check --all-files
```

**Step 5: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "ci: add pre-commit hook for poetry lock validation"
```

---

## Task 6: Update Apollo Similarly (Parallel Work)

This task should be done in the apollo repo, following the same pattern:

1. Create `pyproject.local.toml.example` with paths for:
   - `logos-foundry`
   - `logos-sophia-sdk`
   - `logos-hermes-sdk`
2. Create `scripts/setup-local-dev.sh`
3. Pin git dependencies to tags
4. Add `docs/DEPENDENCY_TROUBLESHOOTING.md`
5. Add `constraints.txt` for CI

**Note:** This task is tracked separately - create issue in apollo repo.

---

## Task 7: Document Cross-Repo Dev Workflow

**Files:**
- Create: `docs/CROSS_REPO_DEVELOPMENT.md`

**Step 1: Write cross-repo guide**

Create `docs/CROSS_REPO_DEVELOPMENT.md`:

```markdown
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
```

**Step 2: Commit**

```bash
git add docs/CROSS_REPO_DEVELOPMENT.md
git commit -m "docs: add cross-repo development guide

Documents workflow for making coordinated changes across
logos, hermes, apollo, and sophia repositories."
```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] `./scripts/setup-local-dev.sh` works on fresh clone
- [ ] `poetry install -E dev` works without network (if logos cloned locally)
- [ ] `poetry lock --no-update` succeeds
- [ ] CI passes with new constraints.txt
- [ ] Pre-commit hooks pass
- [ ] Cross-repo dev workflow documented

## Rollback Plan

If issues arise:
1. Revert pyproject.toml to `@main` dependency
2. Delete `constraints.txt`
3. Remove pre-commit hooks

The local dev scripts and docs can remain as they're additive.
