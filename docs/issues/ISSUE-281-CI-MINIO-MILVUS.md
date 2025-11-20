# Issue #281 — Fix MinIO/Milvus startup in CI integration tests

## Summary
The `integration-test` job in `.github/workflows/ci.yml` currently fails because the MinIO and Milvus service containers never become healthy. MinIO exits immediately since no `server` command is provided, and Milvus falls back to the tini help message because MinIO (and possibly other required env vars) aren’t ready. This issue tracks hardening the workflow so the storage stack boots reliably before integration tests run.

## Tasks
1. **Update MinIO service definition**
   - Add `command: server /data` (and optionally `--console-address :9090` if needed).
   - Extend the health check with `--health-start-period 30s` to prevent premature failures.
   - Verify logs show MinIO staying up on GitHub-hosted runners.

2. **Update Milvus service definition**
   - Add explicit `command: ["milvus", "run", "standalone"]`.
   - Ensure `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, and `DATA_COORD_ADDRESS` env vars are set alongside `MINIO_ADDRESS`/`ETCD_ENDPOINTS`.
   - Increase the health check start period (e.g., 120s) plus retries so Milvus has time to initialize on slower runners.

3. **Adjust “Wait for services” step**
   - Poll the HTTP health endpoint (`curl http://localhost:9091/healthz`) instead of the gRPC port for Milvus readiness.
   - Increase timeout (e.g., 180s for Milvus, 90s for Neo4j) to reduce flaky failures.

4. **Validate the fix**
   - Push a branch/PR to run the workflow and confirm the integration job brings up MinIO + Milvus successfully.
   - Capture logs for future reference (link them in this issue or in the PR description).

## Acceptance Criteria
- MinIO container shows “Endpoints: http://127.0.0.1:9000 …” in logs and passes health checks.
- Milvus container reaches “Milvus standalone ready” and the job proceeds to the pytest step.
- No transient failures related to MinIO/Milvus startup in consecutive CI runs (at least two green runs suggested).

## Status
- ✅ **Resolved in** `ci.yml`: added explicit commands/env vars for MinIO/Milvus, longer health start periods, and improved wait loops (Nov 20, 2025).
- Next CI run should confirm stability; monitor the integration-test job on pushes to `main`/`develop`.
