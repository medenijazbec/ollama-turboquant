#!/bin/sh
set -eu

REPO_DIR="${REPO_DIR:-/srv/share/ollama-benchmark/ollama-main-turboquant}"
COMPOSE_FILE="${COMPOSE_FILE:-$REPO_DIR/docker-compose.ollama-bench.yml}"
SERVICE="${SERVICE:-ollama-bench}"

SMOKE_MODEL="${SMOKE_MODEL:-qwen3-1.7b:udq4kxl}"
SMOKE_RESULTS_DIR="${SMOKE_RESULTS_DIR:-/results/smoke}"
SMOKE_RESULTS_FULL_DIR="${SMOKE_RESULTS_FULL_DIR:-/results_full/smoke}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$REPO_DIR/results_full/run_logs}"
RUN_LOG_FILE="${RUN_LOG_FILE:-$RUN_LOG_DIR/all_ab_smoke_tests_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$RUN_LOG_DIR"

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$RUN_LOG_FILE"
}

run_step() {
  name="$1"
  shift

  log "START ${name}"
  log "RUN $*"
  if "$@"; then
    log "DONE ${name}"
  else
    rc=$?
    log "FAIL ${name} (exit ${rc})"
    exit "$rc"
  fi
}

log "A/B smoke-test batch starting"
log "compose=${COMPOSE_FILE}"
log "service=${SERVICE}"
log "model=${SMOKE_MODEL}"
log "run_log=${RUN_LOG_FILE}"

run_step "base-minimal" \
  docker compose -f "$COMPOSE_FILE" exec "$SERVICE" sh -lc \
  "MODEL=$SMOKE_MODEL CONTEXT_LADDER=16384 OUTDIR=$SMOKE_RESULTS_DIR /work/scripts/run-base-minimal-test-matrix.sh"

run_step "turbo-minimal" \
  docker compose -f "$COMPOSE_FILE" exec "$SERVICE" sh -lc \
  "MODEL=$SMOKE_MODEL CONTEXT_LADDER=16384 OUTDIR=$SMOKE_RESULTS_DIR /work/scripts/run-turbo-minimal-test-matrix.sh"

run_step "ab-minimal" \
  docker compose -f "$COMPOSE_FILE" exec "$SERVICE" sh -lc \
  "MODEL=$SMOKE_MODEL CONTEXT_LADDER=16384 OUTDIR=$SMOKE_RESULTS_DIR /work/scripts/run-ab-minimal-test-matrix.sh"

run_step "ab-ppl" \
  docker compose -f "$COMPOSE_FILE" exec "$SERVICE" sh -lc \
  "MODEL=$SMOKE_MODEL CHUNK_TOKENS=64 CHUNKS=1 OUTDIR=$SMOKE_RESULTS_DIR /work/scripts/run-ab-ppl.sh"

run_step "ab-agentic" \
  docker compose -f "$COMPOSE_FILE" exec "$SERVICE" sh -lc \
  "MODEL=$SMOKE_MODEL NUM_CTX=8192 OUTDIR=$SMOKE_RESULTS_DIR /work/scripts/run-ab-agentic.sh"

run_step "ab-full-large-context" \
  docker compose -f "$COMPOSE_FILE" exec "$SERVICE" sh -lc \
  "MODEL=$SMOKE_MODEL CONTEXT_LADDER=32768,16384 OUTDIR=$SMOKE_RESULTS_FULL_DIR /work/scripts/run-ab-full-large-context.sh"

log "A/B smoke-test batch complete"
