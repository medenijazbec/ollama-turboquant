#!/bin/sh
set -eu

TURBO_HOST="${TURBO_HOST:-http://127.0.0.1:11439}"
MODEL="${MODEL:-qwen3.5-9b:udq4kxl}"
OUTDIR="${OUTDIR:-/results}"
CONTEXT_LADDER="${CONTEXT_LADDER:-128000,104000,96000,65536,32768,16384}"

mkdir -p "$OUTDIR"

kvstress-bench \
  --hosts "$TURBO_HOST" \
  --host-labels turbo \
  --host-kv-support request \
  --model "$MODEL" \
  --profile large-context \
  --workloads long-context-recall,decode-corruption-guard \
  --kv-modes f16,q8_0,tq35,q8_0/tq35 \
  --fa-modes off \
  --context-ladder "$CONTEXT_LADDER" \
  --warmup 0 \
  --epochs 1 \
  --repeats 1 \
  --progress on \
  --output "$OUTDIR/turbo_minimal_test_matrix.csv" \
  --jsonl-output "$OUTDIR/turbo_minimal_test_matrix.jsonl" \
  --summary-output "$OUTDIR/turbo_minimal_test_matrix.md"
