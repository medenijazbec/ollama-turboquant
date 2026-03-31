#!/bin/sh
set -eu

# Kept at the legacy filename so existing docker exec commands still work.
MODEL="qwen3-30b:instruct2507-udq4kxl"
OUTPUT="/results/test-native-ollama.txt"

mkdir -p /results
: > "$OUTPUT"

export OLLAMA_HOST="http://127.0.0.1:11438"

ollama-bench \
  -model "$MODEL" \
  -epochs 6 \
  -warmup 1 \
  -prompt-tokens 512 \
  -max-tokens 64 \
  -format benchstat \
  -output "$OUTPUT"

cat "$OUTPUT"
