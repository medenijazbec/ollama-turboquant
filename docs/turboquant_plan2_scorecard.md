# TurboQuant Plan 2 Scorecard

These scores are static Plan 2 scores only. They do not include generation benchmarks, quality drift, latency measurements, memory benchmarks, or end-to-end model comparisons. Those runtime comparisons remain deferred until Plan 5.

## Score Snapshot

| Metric | Pre | Post | Delta |
| --- | ---: | ---: | ---: |
| Mode Plumbing Completeness Score | 0 | 100 | +100 |
| Static Compatibility Score | 50 | 100 | +50 |

## Pre Snapshot

Pre refers to the branch state before this Plan 2 patch.

- Mode Plumbing Completeness Score: `0/100`
- Static Compatibility Score: `50/100`

### Main pre gaps

- `envconfig/config.go` accepted the modes in practice but the user-facing `OLLAMA_KV_CACHE_TYPE` description did not document `tq25`, `tq35`, `tq3`, or `tq4`
- `llm/server.go` normalized aliases but did not expose the effective mode through a dedicated helper and stable acceptance logging
- `llm/server_test.go` had no Plan 2-specific KV-cache mode coverage
- `fs/ggml/ggml_test.go` covered support and bytes-per-element but not explicit `KVCacheTypeIsQuantized` coverage
- `runner/ollamarunner/cache_test.go` covered TurboQuant aliases but did not explicitly guard `""`, `f16`, `q8_0`, and `q4_0`
- `llama/llama_test.go` verified rejection, but not the error contents or normalized `tq35` alias behavior
- No Plan 2 validation script, score script, or scorecard artifact existed

### Pre strengths

- Alias normalization to `tq35` was already consistent across:
  - `llm/server.go`
  - `fs/ggml/ggml.go`
  - `runner/ollamarunner/cache.go`
  - `llama/llama.go`
- Legacy llama rejection for TurboQuant modes was already explicit

## Post Snapshot

Post refers to the current branch after the Plan 2 patch.

- Mode Plumbing Completeness Score: `100/100`
- Static Compatibility Score: `100/100`

### Post state

- `OLLAMA_KV_CACHE_TYPE` help text now documents `f16`, `q8_0`, `q4_0`, `tq25`, `tq35`, and aliases `tq3` / `tq4`
- `llm/server.go` now exposes explicit internal helpers for:
  - requested versus effective mode normalization
  - legacy-path selection
  - new-engine selection
  - stable acceptance logging
- Tests now cover:
  - canonical and alias normalization
  - flash-attention gating
  - dtype mapping for old and new modes
  - quantized classification in `fs/ggml`
  - explicit legacy rejection error content
- Static Plan 2 validation and scoring scripts now exist
- Runtime pre/post comparisons remain explicitly deferred until Plan 5

## Verification Commands

```powershell
powershell -File .\scripts\score_turboquant_plan2.ps1
powershell -File .\scripts\test_turboquant_plan2.ps1
```

The intended final post-validation result is:

- `Mode Plumbing Completeness Score: 100/100`
- `Static Compatibility Score: 100/100`
- `TurboQuant Plan 2 validation passed.`
