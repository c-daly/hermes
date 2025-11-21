# Copilot Instructions — Hermes

Short, focused guidance for AI coding agents working in `hermes/` (language & embedding utility).

Big picture
- Hermes is a stateless language & embedding microservice used by `apollo` and `sophia`. It provides STT/TTS, lightweight NLP helpers, and text embeddings. The canonical API contract for language/embedding endpoints is tracked in `logos/contracts/hermes.openapi.yaml`.

Key files & locations
- `pyproject.toml` — Python package metadata and dev scripts
- `Dockerfile*`, `docker-compose*.yml` — containerized development and test setups
- `tests/` — unit & integration tests
- `README.md` — repo-specific docs and quick start

Developer workflows
- Install dev deps: `pip install -e ".[dev]"` (run from `hermes/`).
- Run tests: `pytest` (unit tests under `tests/`).
- Run service locally (typical): use the container or `uvicorn` if a FastAPI app is present. Prefer the Docker setup defined in repo for reproducible envs.
- Lint & format: follow `pyproject.toml` settings (`ruff`, `black`, `mypy`) and run them before opening PRs.

Integration & patterns
- Hermes is stateless — do not modify HCG/Neo4j/Milvus here. Expose clear, versioned REST endpoints matching `logos/contracts/hermes.openapi.yaml`.
- Provide deterministic responses for embedding endpoints when running `DEV` or `MOCK` mode to aid offline UI work (`apollo/webapp` uses mock-mode fixtures).

GitHub, tickets & PRs
- Follow the workspace-wide rules in `logos/.github/copilot-instructions.md` for issue titles, labels, branch naming, and PR requirements. Every PR must reference its issue (use `Closes #<n>`).
- When you start work on an issue that lives on the LOGOS workspace project, move its card to *In Progress* (and adjust any `status/*` label). When the work is merged/done, move the card to *Done* so the board stays in sync.

Examples
- Add a new embedding endpoint: update OpenAPI in `logos/contracts/hermes.openapi.yaml`, add server handler, update `tests/` with request/response pairs, and add an integration smoke test.

If anything repo-specific is missing (start commands, env vars), tell me and I will extract exact commands from `pyproject.toml` or `Dockerfile` and update this file.
