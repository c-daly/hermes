"""OpenAPI conformance tests for the Hermes FastAPI app.

These tests validate the *served* OpenAPI document (``app.openapi()``) entirely
in-process via the FastAPI/Starlette route table and ``TestClient`` — no Neo4j,
Milvus, or any external service is required.

What is asserted here:

* **Completeness** — every public route implemented by the app is present in the
  served OpenAPI schema with matching HTTP methods. If a new public route is
  added but accidentally hidden from the schema (``include_in_schema=False``),
  these tests fail.
* **Structural validity** — the served schema is a well-formed OpenAPI 3.x
  document: it declares a version and ``info``, every operation has at least one
  documented response, and there are no duplicate ``operationId`` values.
* **Canonical contract gap** — the served schema is compared against the set of
  paths documented in the canonical contract
  (``logos/contracts/hermes.openapi.yaml``). The app implements several routes
  that are not yet in that contract; that gap is recorded as an expected failure
  pending the contract being expanded (see ``logos#91`` follow-up).

The source of truth for the canonical contract is:
https://github.com/c-daly/logos/blob/main/contracts/hermes.openapi.yaml
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Set, Tuple

import pytest
from starlette.routing import Route

from hermes.main import app

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Canonical contract baseline
# ---------------------------------------------------------------------------
# Paths currently documented in logos/contracts/hermes.openapi.yaml. Kept here
# as an explicit baseline so this test has no dependency on the (un-packaged)
# contract file or network access. Update this set when the contract changes.
CANONICAL_CONTRACT_PATHS: Set[str] = {
    "/stt",
    "/tts",
    "/simple_nlp",
    "/embed_text",
    "/llm",
}

# Routes that are part of the served app but are intentionally NOT part of the
# documented public API surface (infra / tooling / static assets). These are
# excluded from the completeness check.
NON_API_PATHS: Set[str] = {
    "/",
    "/ui",
    "/health",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}

# Routes implemented by the app that are public API but not yet documented in
# the canonical contract. These are the gap this ticket surfaces.
EXPECTED_UNDOCUMENTED_IN_CONTRACT: Set[str] = {
    "/embed_visual",
    "/ingest/media",
    "/feedback",
    "/name-type",
    "/name-relationship",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _implemented_routes() -> Dict[str, Set[str]]:
    """Return ``{path: {METHOD, ...}}`` for every concrete app route.

    HEAD and OPTIONS are dropped because they are auto-added by the framework
    and are not meaningful API operations.
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


def _schema_path_methods(schema: Dict) -> Dict[str, Set[str]]:
    """Return ``{path: {METHOD, ...}}`` documented in an OpenAPI ``paths`` block."""
    result: Dict[str, Set[str]] = {}
    for path, item in schema.get("paths", {}).items():
        result[path] = {method.upper() for method in item}
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openapi_schema() -> Dict:
    """The served OpenAPI document, regenerated fresh for the test session."""
    app.openapi_schema = None  # force regeneration, ignore any cached schema
    return app.openapi()


# ---------------------------------------------------------------------------
# Completeness: every implemented public route is documented in the served schema
# ---------------------------------------------------------------------------


def test_served_schema_documents_every_public_route(openapi_schema: Dict) -> None:
    """Fail if any implemented public route is missing from the served schema.

    This is the core acceptance criterion: a route that exists in the app but is
    not present in ``app.openapi()`` (e.g. accidentally hidden with
    ``include_in_schema=False``) is an undocumented endpoint and is rejected.
    """
    documented = _schema_path_methods(openapi_schema)
    implemented = _public_api_routes()

    missing: list[Tuple[str, Set[str]]] = []
    for path, methods in sorted(implemented.items()):
        documented_methods = documented.get(path, set())
        missing_methods = methods - documented_methods
        if missing_methods:
            missing.append((path, missing_methods))

    assert not missing, (
        "Implemented public route(s) missing from the served OpenAPI schema: "
        + ", ".join(f"{p} {sorted(m)}" for p, m in missing)
    )


@pytest.mark.parametrize("path", sorted(EXPECTED_UNDOCUMENTED_IN_CONTRACT))
def test_previously_undocumented_routes_now_in_served_schema(
    openapi_schema: Dict, path: str
) -> None:
    """Routes that were missing from the contract must at least be self-documented.

    These endpoints (``/embed_visual``, ``/ingest/media``, ``/feedback``,
    ``/name-type``, ``/name-relationship``) are implemented in ``main.py``. They
    must appear in the served schema even while the canonical contract catches
    up, so clients reading ``/openapi.json`` see them.
    """
    documented = _schema_path_methods(openapi_schema)
    assert path in documented, f"{path} is implemented but absent from app.openapi()"
    assert documented[path], f"{path} is in the schema but documents no operations"


def test_canonical_contract_paths_are_served(openapi_schema: Dict) -> None:
    """Every path in the canonical contract must still be served by the app."""
    documented = _schema_path_methods(openapi_schema)
    missing = sorted(CANONICAL_CONTRACT_PATHS - set(documented))
    assert (
        not missing
    ), "Canonical contract path(s) not served by the app (regression): " + ", ".join(
        missing
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
            responses = operation.get("responses") or {}
            if not responses:
                offenders.append(f"{method.upper()} {path}")
    assert not offenders, "Operations without documented responses: " + ", ".join(
        offenders
    )


def test_no_duplicate_operation_ids(openapi_schema: Dict) -> None:
    """``operationId`` values must be unique across the served schema.

    A duplicate ``operationId`` produces an invalid OpenAPI document and breaks
    client/code generators. This previously occurred for the ``/health`` route,
    which registered GET and HEAD under one ``api_route`` (shared operationId).
    """
    op_ids = [
        operation.get("operationId")
        for item in openapi_schema["paths"].values()
        for operation in item.values()
        if operation.get("operationId") is not None
    ]
    duplicates = {op_id: count for op_id, count in Counter(op_ids).items() if count > 1}
    assert not duplicates, f"Duplicate operationId(s) in served schema: {duplicates}"


# ---------------------------------------------------------------------------
# Canonical contract gap (recorded, not yet closed)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "logos/contracts/hermes.openapi.yaml documents only 5 of the implemented "
        "public routes. Expanding the canonical contract to cover the remaining "
        "endpoints is tracked as a follow-up to logos#91."
    ),
    strict=False,
)
def test_canonical_contract_covers_all_public_routes() -> None:
    """The canonical contract should eventually document every public route.

    This intentionally xfails today to record the known gap. When the contract is
    expanded (and ``CANONICAL_CONTRACT_PATHS`` updated to match), it will XPASS,
    signalling the gap is closed and the marker can be removed.
    """
    public_paths = set(_public_api_routes())
    undocumented = sorted(public_paths - CANONICAL_CONTRACT_PATHS)
    assert (
        not undocumented
    ), "Public routes not present in the canonical contract: " + ", ".join(undocumented)
