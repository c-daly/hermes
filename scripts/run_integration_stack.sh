#!/usr/bin/env bash
set -euo pipefail

# Determine repo root: use HERMES_REPO_ROOT if set, otherwise compute from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERMES_REPO_ROOT="${HERMES_REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
export HERMES_REPO_ROOT

# Standard stack location (matches LOGOS layout)
STACK_DIR="${HERMES_REPO_ROOT}/tests/e2e/stack/hermes"
ENV_FILE="${STACK_DIR}/.env.test"
COMPOSE_FILE="${STACK_DIR}/docker-compose.test.yml"

# Load environment from .env.test
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "$ENV_FILE"
  set +a
fi

COMPOSE=${COMPOSE_CMD:-"docker compose"}
# Hermes only needs Milvus (no Neo4j)
SERVICES=("milvus-etcd" "milvus-minio" "milvus")
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-180}

# Hermes-specific ports (17530, 17091 to avoid conflicts with other repos)
PORTS_TO_CHECK=(
  "17530:Milvus gRPC"
  "17091:Milvus health"
)

info() {
  echo "[info] $1"
}

warn() {
  echo "[warn] $1"
}

error() {
  echo "[error] $1"
}

check_port_in_use() {
  local port=$1
  if command -v ss >/dev/null 2>&1; then
    if ss -tulpn 2>/dev/null | grep -q ":${port} "; then
      return 0
    fi
  elif command -v lsof >/dev/null 2>&1; then
    if lsof -i ":${port}" >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

container_id() {
  local service=$1
  local id=""
  id=$($COMPOSE -f "$COMPOSE_FILE" ps -q "$service" 2>/dev/null | head -n1 || true)
  echo "$id"
}

container_display_name() {
  local container=$1
  local name=""
  name=$(docker inspect -f '{{.Name}}' "$container" 2>/dev/null | sed 's#^/##' || true)
  echo "${name:-$container}"
}

wait_for_container() {
  local service=$1
  local container_id=$2
  local display_name=${3:-$2}
  local deadline=$((SECONDS + HEALTH_TIMEOUT))

  while (( SECONDS < deadline )); do
    local status=""
    status=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id" 2>/dev/null || true)

    case "$status" in
      healthy)
        info "$service ($display_name) is healthy"
        return 0
        ;;
      unhealthy)
        error "$service ($display_name) reported unhealthy"
        docker logs "$container_id" --tail=200 || true
        return 1
        ;;
      starting|"" )
        info "$service ($display_name) still starting (status: ${status:-unknown})"
        ;;
      *)
        warn "$service ($display_name) status: $status"
        ;;
    esac
    sleep 5
  done

  error "$service ($display_name) did not become healthy within ${HEALTH_TIMEOUT}s"
  docker logs "$container_id" --tail=200 || true
  return 1
}

cleanup() {
  echo "Stopping integration services..."
  $COMPOSE -f "$COMPOSE_FILE" down -v >/dev/null 2>&1 || true
}

trap cleanup EXIT

echo "Checking for conflicting ports before starting services..."
for mapping in "${PORTS_TO_CHECK[@]}"; do
  port=${mapping%%:*}
  label=${mapping#*:}
  if check_port_in_use "$port"; then
    warn "$label (port $port) already in use; existing process may interfere"
  else
    info "$label port $port is free"
  fi

done

echo "Starting Milvus stack for Hermes integration tests..."
if ! $COMPOSE -f "$COMPOSE_FILE" up -d "${SERVICES[@]}"; then
  error "docker compose failed to start services"
  $COMPOSE -f "$COMPOSE_FILE" logs --tail=200 || true
  exit 1
fi

for service in "${SERVICES[@]}"; do
  container=$(container_id "$service")
  if [[ -z "$container" ]]; then
    error "Unable to determine container ID for service '$service'. Is it running?"
    $COMPOSE -f "$COMPOSE_FILE" ps "$service" || true
    exit 1
  fi
  display_name=$(container_display_name "$container")
  if ! wait_for_container "$service" "$container" "$display_name"; then
    error "Aborting due to unhealthy service: $service"
    exit 1
  fi

done

# Export standard env vars (from .env.test, with localhost for host access)
export MILVUS_HOST=${MILVUS_HOST:-"localhost"}
export MILVUS_PORT=${MILVUS_PORT:-"17530"}
export NEO4J_URI=${NEO4J_URI:-"bolt://localhost:17687"}
export NEO4J_USER=${NEO4J_USER:-"neo4j"}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-"neo4jtest"}
export RUN_HERMES_INTEGRATION=1

default_pytest_args=("tests/test_milvus_integration.py" "-v")
if [[ $# -gt 0 ]]; then
  pytest_args=("$@")
else
  pytest_args=("${default_pytest_args[@]}")
fi

info "Running Hermes integration tests via pytest ${pytest_args[*]}"
poetry run pytest "${pytest_args[@]}"
