# Agentic and Recall Benchmarks

`kvstress-bench` now includes deterministic correctness-oriented workloads:

- `long-context-recall`
- `prompt-file-regression`
- `decode-corruption-guard`
- `agentic-structured`

The recall suite is intended to catch context-pressure failures that plain speed or PPL numbers can miss. The structured workload uses schema-constrained mock tool-style prompts so rows are only treated as valid when the output stays well-formed and semantically correct.

Minimal agentic A/B run:

```bash
docker compose -f docker-compose.ollama-bench.yml exec ollama-bench \
  /scripts/run-ab-agentic.sh
```

The default minimal correctness model is `qwen3.5-9b:udq4kxl`. The default small distill correctness model is `deepseek-r1-distill-qwen7:q4km`.
