# Intake: Hermes #50 + #26 Standardization

## What
Adopt shared standardization from logos_test_utils: config helper, health schema (from logos_config), logging setup, and add HEAD method to /health endpoint.

## Why
Ensure consistent behavior across LOGOS stack (Sophia already standardized in Phase 1). Enable proper health checks for Apollo SDK integration (#26 unblocks HermesClient.health_check()).

## Success Criteria
1. No import-time env loads (verify by importing main module with missing env vars)
2. `/health` returns `logos_config.health.HealthResponse` JSON with version, dependencies
3. `/health` supports HEAD method (returns 200 OK without body)
4. Logs are JSON with timestamp/level/logger/message (default) or human-readable with `LOG_FORMAT=human`
5. Request ID middleware adds X-Request-ID header
6. All existing tests pass

## Constraints
- Follow sophia standardization pattern exactly
- Use `logos_config.health.HealthResponse` and `DependencyStatus` (already available via logos_config)
- Use `logos_test_utils.setup_logging` (need to add dependency)
- Do NOT break existing functionality

## Files Affected
| File | Action | Changes |
|------|--------|---------|
| `pyproject.toml` | Modify | Add logos_test_utils dependency |
| `src/hermes/main.py` | Modify | Replace logging, add middleware, update /health, add HEAD |
| `src/hermes/milvus_client.py` | Modify | Make config lazy (remove module-level env reads) |
| `src/hermes/env.py` | Review | May remove if superseded by logos_test_utils |

## Relevant Capabilities
- **logos_config.health**: `HealthResponse`, `DependencyStatus` schemas
- **logos_test_utils**: `setup_logging()` for structured logging
- **Sophia pattern**: `RequestIDMiddleware` implementation to copy

## Workflow
**Classification:** Complex (multi-file, cross-repo pattern adoption)

## Dependencies
- logos #434 (config helper) ✅ merged
- logos #435 (health/logging schemas) ✅ merged
