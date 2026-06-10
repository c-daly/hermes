"""OpenAPI conformance gate for the Hermes FastAPI app.

The served OpenAPI document (``app.openapi()``) is the **source of truth** for
the Hermes API contract. ``openapi.yaml`` at the repo root is a snapshot of that
document, regenerated with ``make openapi`` (``python scripts/export_openapi.py``)
and pulled by logos into ``contracts/hermes.openapi.yaml`` to drive SDK and docs
generation.

These tests are a **gate, not a record**: if a route is added, removed, or
renamed — or an operation's schema changes — and the snapshot is not refreshed,
CI fails. This replaces the previous hand-maintained ``CANONICAL_CONTRACT_PATHS``
baseline and the ``xfail`` that merely recorded contract drift. Drift can no
longer merge silently.

All assertions run in-process via the FastAPI/Starlette route table; no Neo4j,
Milvus, or other external service is required.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Generator, Set, Tuple

import pytest
import yaml
from starlette.routing import Route

from hermes.main import app

pytestmark = pytest.mark.unit

# openapi.yaml lives at the repo root: tests/unit/<this file> -> parents[2].
SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "openapi.yaml"

# Infra / tooling / static routes that are not part of the public API surface
# and need no contract documentation.
NON_API_PATHS: Set[str] = {
    "/",
    "/ui",
    "/health",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}

# An OpenAPI Path Item Object may hold non-operation keys (summary, parameters,
# $ref, ...) alongside HTTP operations. Only these are operations.
HTTP_METHODS = frozenset(
    {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _implemented_routes() -> Dict[str, Set[str]]:
    """Return ``{path: {METHOD, ...}}`` for every concrete app route.

    HEAD and OPTIONS are dropped: the framework adds them and they are not
    meaningful API operations.
    """
    routes: Dict[str, Set[str]] = {}
    for route in app.routes:
        if not isinstance(route, Route):
            continue  # Mounts (e.g. /static) are not OpenAPI operations.
        methods = {m for m in (route.methods or set()) if m not in {"HEAD", "OPTIONS"}}
        if not methods:
            continue
        routes.setdefault(route.path, set()).update(methods)
    return routes


def _public_api_routes() -> Dict[str, Set[str]]:
    """Implemented routes minus infra/tooling routes that need no documentation."""
    return {
        path: methods
        for path, methods in _implemented_routes().items()
        if path not in NON_API_PATHS
    }


def _surface(schema: Dict) -> Set[Tuple[str, str]]:
    """Return ``{(path, METHOD), ...}`` for every operation in an OpenAPI schema."""
    out: Set[Tuple[str, str]] = set()
    for path, item in schema.get("paths", {}).items():
        for method in item:
            if method.lower() in HTTP_METHODS:
                out.add((path, method.upper()))
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openapi_schema() -> Generator[Dict, None, None]:
    """The served OpenAPI document, regenerated fresh and restored on teardown."""
    previous = app.openapi_schema
    app.openapi_schema = None  # force regeneration, ignore any cached schema
    try:
        yield app.openapi()
    finally:
        app.openapi_schema = previous


@pytest.fixture(scope="module")
def snapshot_schema() -> Dict:
    """The committed openapi.yaml snapshot, parsed."""
    assert SNAPSHOT_PATH.exists(), (
        f"{SNAPSHOT_PATH.name} is missing — generate it with: make openapi "
        "(python scripts/export_openapi.py)"
    )
    return yaml.safe_load(SNAPSHOT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The gate: served document must equal the committed snapshot
# ---------------------------------------------------------------------------


def test_served_document_matches_snapshot(
    openapi_schema: Dict, snapshot_schema: Dict
) -> None:
    """``app.openapi()`` must match the committed openapi.yaml snapshot.

    This is the core gate. A mismatch means the live app drifted from the
    contract snapshot. Refresh and commit it in the *same* change that altered
    the API: ``make openapi`` (``python scripts/export_openapi.py``).
    """
    if openapi_schema == snapshot_schema:
        return

    served = _surface(openapi_schema)
    snapshot = _surface(snapshot_schema)
    added = sorted(served - snapshot)
    removed = sorted(snapshot - served)

    if added or removed:
        detail = (
            f"\n  routes served but absent from openapi.yaml: {added}"
            f"\n  routes in openapi.yaml but no longer served: {removed}"
        )
    else:
        detail = (
            "\n  (route surface is unchanged — an operation's schema, parameters, "
            "responses, or metadata changed)"
        )

    pytest.fail(
        "Served app.openapi() differs from the committed openapi.yaml snapshot. "
        "Regenerate and commit it in this change: make openapi "
        "(python scripts/export_openapi.py)." + detail
    )


def test_snapshot_documents_every_public_route(openapi_schema: Dict) -> None:
    """Every implemented public route is present in the served schema.

    Catches a route accidentally hidden with ``include_in_schema=False`` (which
    would silently drop it from the snapshot and therefore the SDK).
    """
    documented = _surface(openapi_schema)
    missing: list[str] = []
    for path, methods in sorted(_public_api_routes().items()):
        for method in sorted(methods):
            if (path, method) not in documented:
                missing.append(f"{method} {path}")
    assert not missing, (
        "Implemented public route(s) missing from app.openapi() "
        "(hidden via include_in_schema=False?): " + ", ".join(missing)
    )


# ---------------------------------------------------------------------------
# Structural validity of the served OpenAPI document
# ---------------------------------------------------------------------------


def test_served_schema_is_well_formed(openapi_schema: Dict) -> None:
    """The served document declares an OpenAPI version and basic ``info``."""
    assert str(openapi_schema.get("openapi", "")).startswith(
        "3."
    ), "Served schema must declare an OpenAPI 3.x version"
    info = openapi_schema.get("info", {})
    assert info.get("title"), "OpenAPI info.title must be set"
    assert info.get("version"), "OpenAPI info.version must be set"
    assert openapi_schema.get("paths"), "OpenAPI paths block must be non-empty"


def test_every_operation_documents_responses(openapi_schema: Dict) -> None:
    """Every documented operation declares at least one response."""
    offenders: list[str] = []
    for path, item in openapi_schema["paths"].items():
        for method, operation in item.items():
            if method.lower() not in HTTP_METHODS:
                continue
            if not (operation.get("responses") or {}):
                offenders.append(f"{method.upper()} {path}")
    assert not offenders, "Operations without documented responses: " + ", ".join(
        offenders
    )


def test_no_duplicate_operation_ids(openapi_schema: Dict) -> None:
    """``operationId`` values must be unique across the served schema.

    A duplicate produces an invalid document and breaks client/code generators.
    """
    op_ids = [
        operation.get("operationId")
        for item in openapi_schema["paths"].values()
        for method, operation in item.items()
        if method.lower() in HTTP_METHODS and operation.get("operationId") is not None
    ]
    duplicates = {op_id: count for op_id, count in Counter(op_ids).items() if count > 1}
    assert not duplicates, f"Duplicate operationId(s) in served schema: {duplicates}"


def test_surface_ignores_non_operation_keys() -> None:
    """Path Item non-operation keys (summary, parameters, $ref) are not methods."""
    schema = {
        "paths": {
            "/x": {
                "summary": "x",
                "parameters": [{"name": "q", "in": "query"}],
                "get": {"responses": {"200": {}}},
            }
        }
    }
    assert _surface(schema) == {("/x", "GET")}
