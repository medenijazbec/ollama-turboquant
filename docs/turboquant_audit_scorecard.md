# TurboQuant Audit Scorecard

These scores are audit and static-analysis scores only. They are not runtime quality, latency, memory-benchmark, or model-output scores. Full pre/post runtime comparisons remain deferred until Plan 5.

## Score Snapshot

| Metric | Pre | Post | Delta |
| --- | ---: | ---: | ---: |
| Audit Completeness Score | 64 | 100 | +36 |
| Static Pipeline Coverage Score | 56 | 100 | +44 |

## Pre Snapshot

Pre refers to the audit document that existed before this Plan 1 rewrite.

- Audit Completeness Score: `64/100`
- Static Pipeline Coverage Score: `56/100`

### Main pre gaps

- Missing source-basis references to `ollama_turboquant_full_plan.txt` and `ollama_turboquant_algorithms.txt`
- Missing formal cross-reference matrix
- Missing `runner/ollamarunner/runner.go` in the KV trace
- Missing explicit `Plan 2`, `Plan 3`, `Plan 4`, and `Plan 5` insertion-point handoff
- Missing future fast-path file targets:
  - `ml/backend/ggml/ggml/src/ggml-backend.cpp`
  - `ml/backend/ggml/ggml/src/ggml.c`
  - `ml/backend/ggml/ggml/src/ggml-quants.c`
- Missing required seam references such as `normalizeKVCacheType`, `NewInputCache`, `ScaledDotProductAttention`, `ggml_backend_graph_compute_async`, and `ggml_backend_tensor_set_async`

## Post Snapshot

Post refers to the rewritten `docs/turboquant_audit.md` plus the new validation/scoring scripts.

- Audit Completeness Score: `100/100`
- Static Pipeline Coverage Score: `100/100`

### Post state

- Both TurboQuant TXT source files are cited explicitly
- The cross-reference matrix maps TXT requirements to concrete repo files and symbols
- The `OLLAMA_KV_CACHE_TYPE` trace is complete from `envconfig/config.go` to backend execution seams
- The hard-coded `f16` / `q8_0` / `q4_0` assumptions and KV memory-estimation callsites are enumerated
- Cache contract, tensor shapes, legacy runner behavior, and Phase A versus Phase B boundaries are documented
- Exact insertion points for Plans 2-5 are called out, including `ggml-backend.cpp`, `ggml.c`, and `ggml-quants.c`

## Verification Commands

The Plan 1 artifact was validated with:

```powershell
powershell -File .\scripts\score_turboquant_audit.ps1
powershell -File .\scripts\test_turboquant_audit.ps1
```

The resulting post-validation status was:

- `Audit Completeness Score: 100/100`
- `Static Pipeline Coverage Score: 100/100`
- `TurboQuant audit validation passed.`
