#!/bin/sh
set -eu

BASE_HOST="${BASE_HOST:-http://127.0.0.1:11438}"
TURBO_HOST="${TURBO_HOST:-http://127.0.0.1:11439}"
MODEL="${MODEL:-qwen3.5-9b:udq4kxl}"
OUTDIR="${OUTDIR:-/results}"
CONTEXT_LADDER="${CONTEXT_LADDER:-128000,104000,96000,65536,32768,16384}"

mkdir -p "$OUTDIR"

kvstress-bench \
  --hosts "$BASE_HOST,$TURBO_HOST" \
  --host-labels baseline,turbo \
  --host-kv-support legacy,request \
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
  --output "$OUTDIR/qwen35_9b_minimal.csv" \
  --jsonl-output "$OUTDIR/qwen35_9b_minimal.jsonl" \
  --summary-output "$OUTDIR/qwen35_9b_minimal.md"
