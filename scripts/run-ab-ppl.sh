#!/bin/sh
set -eu

BASE_HOST="${BASE_HOST:-http://127.0.0.1:11438}"
TURBO_HOST="${TURBO_HOST:-http://127.0.0.1:11439}"
MODEL="${MODEL:-qwen3.5-9b:udq4kxl}"
CORPUS="${CORPUS:-/work/cmd/benchppl/testdata/wiki_short.txt}"
OUTDIR="${OUTDIR:-/results}"
CHUNK_TOKENS="${CHUNK_TOKENS:-384}"
CHUNKS="${CHUNKS:-8}"

mkdir -p "$OUTDIR"

ppl-bench \
  --hosts "$BASE_HOST,$TURBO_HOST" \
  --host-labels baseline,turbo \
  --host-kv-support legacy,request \
  --model "$MODEL" \
  --kv-modes f16,q8_0,q4_0,tq25,tq35,q8_0/tq35 \
  --fa-modes off \
  --corpus "$CORPUS" \
  --chunk-tokens "$CHUNK_TOKENS" \
  --chunks "$CHUNKS" \
  --output "$OUTDIR/qwen35_9b_ppl.csv" \
  --jsonl-output "$OUTDIR/qwen35_9b_ppl.jsonl" \
  --summary-output "$OUTDIR/qwen35_9b_ppl.md"
