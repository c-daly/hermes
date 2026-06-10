#!/usr/bin/env python3
"""Export the served OpenAPI document to the repo-local contract snapshot.

This makes the **server the source of truth** for the Hermes API contract: the
snapshot ``openapi.yaml`` (repo root) is generated from ``app.openapi()`` rather
than hand-maintained. It is consumed by:

* ``tests/unit/test_openapi_conformance.py`` — fails CI if the served surface or
  schema drifts from the committed snapshot. Refresh it with ``make openapi``.
* logos ``contracts/`` sync — logos pulls this file into
  ``contracts/hermes.openapi.yaml`` to drive SDK and docs generation.

This replaces the previous hand-maintained workflow (editing the contract by
hand, the orphan ``openapi_patch.yaml`` fragment, and the conformance ``xfail``
that merely *recorded* drift).

Usage::

    python scripts/export_openapi.py            # write openapi.yaml
    python scripts/export_openapi.py --check     # exit 1 if it would change
    python scripts/export_openapi.py --stdout    # print, do not write
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from hermes.main import app

SNAPSHOT = Path(__file__).resolve().parent.parent / "openapi.yaml"


def render() -> str:
    """Return the served OpenAPI document as canonical, diff-stable YAML."""
    schema = app.openapi()
    # sort_keys=True makes regeneration deterministic regardless of the dict
    # insertion order FastAPI happens to emit, so the snapshot only changes when
    # the contract actually changes.
    return yaml.safe_dump(
        schema, sort_keys=True, default_flow_style=False, allow_unicode=True
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the served OpenAPI document to openapi.yaml"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 (without writing) if the snapshot is stale",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="print the document to stdout instead of writing the file",
    )
    args = parser.parse_args(argv)

    rendered = render()

    if args.stdout:
        sys.stdout.write(rendered)
        return 0

    if args.check:
        current = SNAPSHOT.read_text(encoding="utf-8") if SNAPSHOT.exists() else ""
        if current != rendered:
            sys.stderr.write(
                f"{SNAPSHOT.name} is out of date with app.openapi().\n"
                "Regenerate and commit it: make openapi "
                "(python scripts/export_openapi.py)\n"
            )
            return 1
        return 0

    SNAPSHOT.write_text(rendered, encoding="utf-8")
    sys.stdout.write(f"Wrote {SNAPSHOT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
