#!/bin/sh
set -eu

BASE_HOST="${BASE_HOST:-http://127.0.0.1:11438}"
TURBO_HOST="${TURBO_HOST:-http://127.0.0.1:11439}"
MODEL="${MODEL:-qwen3.5-9b:udq4kxl}"
OUTDIR="${OUTDIR:-/results}"
NUM_CTX="${NUM_CTX:-32768}"

mkdir -p "$OUTDIR"

kvstress-bench \
  --hosts "$BASE_HOST,$TURBO_HOST" \
  --host-labels baseline,turbo \
  --host-kv-support legacy,request \
  --model "$MODEL" \
  --profile agentic \
  --workloads agentic-structured \
  --kv-modes f16,tq35,q8_0/tq35 \
  --fa-modes off \
  --num-ctx "$NUM_CTX" \
  --warmup 0 \
  --epochs 1 \
  --repeats 1 \
  --progress on \
  --output "$OUTDIR/qwen35_9b_agentic.csv" \
  --jsonl-output "$OUTDIR/qwen35_9b_agentic.jsonl" \
  --summary-output "$OUTDIR/qwen35_9b_agentic.md"
