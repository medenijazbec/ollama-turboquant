#!/bin/sh
set -eu

BASE_HOST="${BASE_HOST:-http://127.0.0.1:11438}"
MODEL="${MODEL:-qwen3.5-9b:udq4kxl}"
OUTDIR="${OUTDIR:-/results}"

mkdir -p "$OUTDIR"

kvstress-bench \
  --hosts "$BASE_HOST" \
  --host-labels baseline \
  --host-kv-support legacy \
  --model "$MODEL" \
  --profile test-matrix \
  --workloads long-context-recall,decode-corruption-guard,prompt-file-regression \
  --kv-modes f16 \
  --fa-modes off \
  --repeats 3 \
  --output "$OUTDIR/base_minimal_test_matrix.csv" \
  --jsonl-output "$OUTDIR/base_minimal_test_matrix.jsonl" \
  --summary-output "$OUTDIR/base_minimal_test_matrix.md"
