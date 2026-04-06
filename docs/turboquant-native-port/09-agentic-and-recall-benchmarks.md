# Agentic and Recall Benchmarks

`kvstress-bench` now includes deterministic correctness-oriented workloads:

- `long-context-recall`
- `niah-retrieval`
- `prompt-file-regression`
- `decode-corruption-guard`
- `agentic-structured`

The recall suite is intended to catch context-pressure failures that plain speed or PPL numbers can miss. The structured workload uses schema-constrained mock tool-style prompts so rows are only treated as valid when the output stays well-formed and semantically correct.

The validator emphasis in this checkout is:

- semantic correctness
- NIAH exact retrieval pass/fail
- sparse KLD or PPL where the score route supports it
- corruption classification

Token-match is only a secondary proxy in summaries.

Minimal host-shell agentic A/B run:

```bash
bash ./scripts/run-ab-agentic.sh
```

The default minimal correctness model is `qwen3.5-9b:udq4kxl`. The default small distill correctness model is `deepseek-r1-distill-qwen7:q4km`.

Corruption classification now distinguishes:

- `empty_output`
- `punctuation_repetition`
- `invalid_utf8`
- `malformed_json`
- `degenerate_small_alphabet`
- `truncation_detected`
- `verbose_runaway`
