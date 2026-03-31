#!/bin/sh
set -eu

HOSTS="${HOSTS:-http://127.0.0.1:11438,http://127.0.0.1:11439}"
LABELS="${HOST_LABELS:-baseline,turbo}"
MODEL="${MODEL:-qwen3-30b:instruct2507-udq4kxl}"
KVMODES="${KV_MODES:-f16,q8_0,q4_0,tq25,tq35}"
PROFILE="${PROFILE:-full}"
OUTDIR="${OUTDIR:-/results}"
OUTBASE="${OUTBASE:-qwen30b_kvstress}"

mkdir -p "$OUTDIR"

kvstress-bench \
  --hosts "$HOSTS" \
  --host-labels "$LABELS" \
  --model "$MODEL" \
  --kv-modes "$KVMODES" \
  --profile "$PROFILE" \
  --output "$OUTDIR/${OUTBASE}.csv" \
  --jsonl-output "$OUTDIR/${OUTBASE}.jsonl" \
  --summary-output "$OUTDIR/${OUTBASE}.txt"
