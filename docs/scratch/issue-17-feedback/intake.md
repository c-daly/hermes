# Intake: Issue #17 - Receive and Persist Sophia Feedback

## What
Add POST `/feedback` endpoint to receive structured feedback from Sophia about proposal outcomes.

## Why
Sophia emits feedback (SHACL failures, execution results, state diffs) tied to proposal/correlation IDs. Hermes needs to receive and surface this to `/llm` calls so prompt builders can adapt to real-world outcomes.

## Context Gathered

### Sophia Side (Ready)
- **Feedback infrastructure exists**: `src/sophia/feedback/` with queue, worker, dispatcher
- **Payload model defined**: `FeedbackPayload` in sophia with:
  - Correlation keys: `correlation_id`, `plan_id`, `execution_id` (at least one required)
  - `feedback_type`: observation | plan | execution | validation
  - `outcome`: accepted | rejected | created | success | failure | partial
  - `reason`: string explanation
  - Optional: `state_diff`, `step_results`, `node_ids_created`
- **Worker sends to**: `{hermes_url}/feedback`

### Hermes Side (Current State)
- **Stateless by design**: "Milvus writes are the only persistence (for embeddings)"
- **Auth pattern**: Bearer tokens from env vars (`SOPHIA_API_TOKEN` for outbound)
- **Routes in**: `src/hermes/main.py` (FastAPI app)
- **Existing Sophia integration**: `_forward_llm_to_sophia()` sends proposals to Sophia

## Architectural Decisions Needed

### 1. Storage Approach
**Constraint**: Hermes must remain stateless per AGENTS.md

| Option | Pros | Cons |
|--------|------|------|
| In-memory TTL cache | Simple, no deps, fits "recent" use case | Lost on restart |
| Milvus (allowed) | Persistent, already integrated | Overkill for simple key-value |
| Redis | Durable, TTL native | New dependency |

**Recommendation**: In-memory TTL cache (1-hour default). Feedback is ephemeral context - losing it on restart is acceptable for "recent" feedback queries.

### 2. Auth Mechanism
- Sophia already uses `SOPHIA_API_TOKEN` for outbound
- Hermes needs `HERMES_API_TOKEN` (or similar) for inbound
- Simple Bearer token validation in middleware

### 3. Rate Limiting
- Simple in-memory counter per source IP or token
- No new dependencies needed

## Success Criteria
- [ ] POST `/feedback` accepts `FeedbackPayload` format
- [ ] Invalid payloads return 4xx with actionable errors
- [ ] Auth required (Bearer token)
- [ ] Feedback queryable by correlation_id/plan_id/execution_id
- [ ] `/llm` can optionally pull recent feedback for context
- [ ] Structured logs for ingestion success/failure
- [ ] Unit tests with mocked payloads
- [ ] Integration test with Sophia (if stack available)

## Constraints
- Maintain statelessness (in-memory cache acceptable)
- Follow existing patterns in `main.py`
- Use Bearer token auth (env-configured)
- No new heavy dependencies

## Relevant Capabilities
- Serena symbolic tools for code exploration/editing
- Context7 for FastAPI docs if needed
- pytest for testing

## Workflow
**Classification**: COMPLEX (multi-file, architectural decisions, new endpoint + storage + auth)

## Open Questions for User
1. **Storage**: Is in-memory TTL cache acceptable, or do you need persistence across restarts?
2. **TTL**: Default 1 hour for feedback retention - is this appropriate?
3. **Auth**: Use `HERMES_API_TOKEN` env var for inbound auth?
