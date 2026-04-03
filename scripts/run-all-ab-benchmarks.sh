#!/bin/sh
set -eu

REPO_DIR="${REPO_DIR:-/srv/share/ollama-benchmark/ollama-main-turboquant}"
COMPOSE_FILE="${COMPOSE_FILE:-$REPO_DIR/docker-compose.ollama-bench.yml}"
SERVICE="${SERVICE:-ollama-bench}"

RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results}"
RESULTS_FULL_DIR="${RESULTS_FULL_DIR:-$REPO_DIR/results_full}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$RESULTS_FULL_DIR/run_logs}"
RUN_LOG_FILE="${RUN_LOG_FILE:-$RUN_LOG_DIR/all_ab_benchmarks_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$RESULTS_DIR" "$RESULTS_FULL_DIR" "$RUN_LOG_DIR"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RUN_LOG_FILE"
}

run_step() {
  name="$1"
  script_path="$2"

  log "START ${name}"
  log "RUN docker compose -f ${COMPOSE_FILE} exec ${SERVICE} sh ${script_path}"

  if docker compose -f "$COMPOSE_FILE" exec "$SERVICE" sh "$script_path"; then
    log "DONE ${name}"
  else
    rc=$?
    log "FAIL ${name} (exit ${rc})"
    exit "$rc"
  fi
}

log "A/B benchmark batch starting"
log "repo=${REPO_DIR}"
log "compose=${COMPOSE_FILE}"
log "service=${SERVICE}"
log "results=${RESULTS_DIR}"
log "results_full=${RESULTS_FULL_DIR}"
log "run_log=${RUN_LOG_FILE}"

if ! docker compose -f "$COMPOSE_FILE" ps "$SERVICE" >/dev/null 2>&1; then
  log "FAIL ${SERVICE} is not available via ${COMPOSE_FILE}"
  exit 1
fi

run_step "base-minimal-test-matrix" "/work/scripts/run-base-minimal-test-matrix.sh"
run_step "turbo-minimal-test-matrix" "/work/scripts/run-turbo-minimal-test-matrix.sh"
run_step "ab-minimal-test-matrix" "/work/scripts/run-ab-minimal-test-matrix.sh"
run_step "ab-ppl" "/work/scripts/run-ab-ppl.sh"
run_step "ab-agentic" "/work/scripts/run-ab-agentic.sh"
run_step "ab-full-large-context" "/work/scripts/run-ab-full-large-context.sh"

log "A/B benchmark batch complete"
