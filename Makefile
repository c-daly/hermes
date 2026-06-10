.PHONY: openapi openapi-check

# Regenerate the OpenAPI contract snapshot (openapi.yaml) from the live app.
# Run this whenever you add, remove, rename, or change the schema of a route,
# and commit the result in the same change.
openapi:
	poetry run python scripts/export_openapi.py

# Fail if the committed snapshot is stale. Wire this into CI as a fast guard in
# addition to the conformance test in tests/unit/test_openapi_conformance.py.
openapi-check:
	poetry run python scripts/export_openapi.py --check
