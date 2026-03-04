<!--
Pull Request template for Project LOGOS
Ensure the PR references its issue (use `Closes #<issue-number>` when ready to close).
-->

# PR Title
[area] Short summary (#<issue>)

## Summary
Brief description of the change and why it is needed. Keep it short and focused.

## Related Issue
Closes #<issue-number>
Or: See #<issue-number> (explain partial coverage)

## Changes
- Files / modules changed:
  - `path/to/file.py`
  - `webapp/src/components/MyPanel.tsx`

## How to run / test locally
Include exact commands and environment setup (copyable):

```bash
# From repo root
pip install -e ".[dev]"
cd webapp && npm install
# Start mock webapp
cd webapp && npm run dev:mock
# Run Python tests
pytest
```

## Checklist (required)
- [ ] Linked the related issue (`Closes #<n>` or `See #<n>`)
- [ ] Tests added or updated
- [ ] Linting/formatting run (`black`/`ruff`; `npm run lint` for webapp)
- [ ] Type checks (Python `mypy`/TS `npm run type-check`) as applicable
- [ ] Documentation updated (README, `docs/`, or OpenAPI spec)
- [ ] CI is passing

## Reviewers / Code owners
Please request at least one reviewer from the relevant area (use `CODEOWNERS` if present).

## Notes
Any additional notes for reviewers, rationale, or things to watch for.
