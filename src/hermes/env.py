"""Environment helpers for Hermes configuration and testing.

This module provides utilities for resolving paths and loading environment
configuration, following the pattern established in logos_test_utils.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import cache
from pathlib import Path

from logos_config.ports import get_repo_ports

# Get hermes port defaults from centralized config
_HERMES_PORTS = get_repo_ports("hermes")


def get_env_value(
    key: str,
    env: Mapping[str, str] | None = None,
    default: str | None = None,
) -> str | None:
    """Resolve an env var by checking OS env, provided mapping, then default.

    Args:
        key: Environment variable name
        env: Optional mapping to check (e.g., loaded from .env file)
        default: Default value if not found

    Returns:
        The resolved value or default
    """
    if key in os.environ:
        return os.environ[key]
    if env and key in env:
        return env[key]
    return default


def get_repo_root(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the Hermes repo root, honoring HERMES_REPO_ROOT if set.

    Priority:
    1. HERMES_REPO_ROOT from OS env or provided mapping (if path exists).
    2. GITHUB_WORKSPACE (set by GitHub Actions in CI).
    3. Fallback to parent of this package (works when running from source).

    Args:
        env: Optional mapping to check for HERMES_REPO_ROOT

    Returns:
        Path to the repository root
    """
    env_value = get_env_value("HERMES_REPO_ROOT", env)
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if candidate.exists():
            return candidate

    # GitHub Actions sets GITHUB_WORKSPACE to the repo checkout
    github_workspace = os.getenv("GITHUB_WORKSPACE")
    if github_workspace:
        candidate = Path(github_workspace).resolve()
        if candidate.exists():
            return candidate

    # Fallback: this file is at src/hermes/env.py, so parents[2] is repo root
    return Path(__file__).resolve().parents[2]


@cache
def load_env_file(env_path: str | Path | None = None) -> dict[str, str]:
    """Load environment variables from a .env file.

    Args:
        env_path: Path to .env file. If None, tries .env.test in repo root.

    Returns:
        Dictionary of environment variables
    """
    if env_path is None:
        repo_root = get_repo_root()
        env_path = repo_root / ".env.test"

    path = Path(env_path)
    env: dict[str, str] = {}

    if not path.exists():
        return env

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        # Strip quotes from values
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        env[key.strip()] = value

    return env


# Service connection configuration helpers


def get_milvus_config(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Get Milvus connection configuration from environment.

    Args:
        env: Optional mapping to check for values

    Returns:
        Dictionary with host, port, and collection_name
    """
    # These all have defaults so they won't be None
    host = get_env_value("MILVUS_HOST", env, "localhost")
    port = get_env_value("MILVUS_PORT", env, str(_HERMES_PORTS.milvus_grpc))
    collection_name = get_env_value("MILVUS_COLLECTION_NAME", env, "hermes_embeddings")
    assert host is not None
    assert port is not None
    assert collection_name is not None
    return {
        "host": host,
        "port": port,
        "collection_name": collection_name,
    }


def get_neo4j_config(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Get Neo4j connection configuration from environment.

    Args:
        env: Optional mapping to check for values

    Returns:
        Dictionary with uri, user, and password
    """
    # These all have defaults so they won't be None
    uri = get_env_value(
        "NEO4J_URI", env, f"bolt://localhost:{_HERMES_PORTS.neo4j_bolt}"
    )
    user = get_env_value("NEO4J_USER", env, "neo4j")
    password = get_env_value("NEO4J_PASSWORD", env, "password")
    assert uri is not None
    assert user is not None
    assert password is not None
    return {
        "uri": uri,
        "user": user,
        "password": password,
    }
