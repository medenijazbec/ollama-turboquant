#!/bin/sh
set -eu

BASE_HOST="${BASE_HOST:-http://127.0.0.1:11438}"
TURBO_HOST="${TURBO_HOST:-http://127.0.0.1:11439}"
MODEL="${MODEL:-qwen35:udiq4xs}"
OUTDIR="${OUTDIR:-/results_full}"
CONTEXT_LADDER="${CONTEXT_LADDER:-262144,128000,104000,96000,65536,32768,16384}"

mkdir -p "$OUTDIR"

kvstress-bench \
  --hosts "$BASE_HOST,$TURBO_HOST" \
  --host-labels baseline,turbo \
  --host-kv-support legacy,request \
  --model "$MODEL" \
  --profile large-context \
  --workloads fit-ceiling,long-context-recall \
  --kv-modes f16,q8_0,tq35 \
  --fa-modes off \
  --context-ladder "$CONTEXT_LADDER" \
  --warmup 0 \
  --epochs 1 \
  --repeats 1 \
  --progress on \
  --output "$OUTDIR/qwen35_udiq4xs_large_context.csv" \
  --jsonl-output "$OUTDIR/qwen35_udiq4xs_large_context.jsonl" \
  --summary-output "$OUTDIR/qwen35_udiq4xs_large_context.md"
