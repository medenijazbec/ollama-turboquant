#!/bin/sh
set -eu

TURBO_HOST="${TURBO_HOST:-http://127.0.0.1:11439}"
MODEL="${MODEL:-qwen3.5-9b:udq4kxl}"
OUTDIR="${OUTDIR:-/results}"

mkdir -p "$OUTDIR"

kvstress-bench \
  --hosts "$TURBO_HOST" \
  --host-labels turbo \
  --host-kv-support request \
  --model "$MODEL" \
  --profile test-matrix \
  --workloads long-context-recall,decode-corruption-guard,prompt-file-regression \
  --kv-modes f16,q8_0,q4_0,tq25,tq35,q8_0/tq35 \
  --fa-modes both \
  --repeats 3 \
  --output "$OUTDIR/turbo_minimal_test_matrix.csv" \
  --jsonl-output "$OUTDIR/turbo_minimal_test_matrix.jsonl" \
  --summary-output "$OUTDIR/turbo_minimal_test_matrix.md"
