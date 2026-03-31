# TurboQuant Completion Matrix

This matrix cross-references the implemented TurboQuant branch against:

- `01_kv_pipeline_audit.plan.txt`
- `02_mode_plumbing_and_memory.plan.txt`
- `03_turboquant_core_codec.plan.txt`
- `04_kvcache_phase_a_integration.plan.txt`
- `05_backend_fastpath_and_validation.plan.txt`
- `ollama_turboquant_full_plan.txt`
- `ollama_turboquant_algorithms.txt`

## Matrix

| Requirement Source | Requirement | Repo File(s) | Symbol / Artifact | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| Plan 1 | KV pipeline audit | `docs/turboquant_audit.md` | audit doc, scorecard, scripts | Complete | Branch drift called out explicitly |
| Plan 2 | KV mode plumbing | `envconfig/config.go`, `llm/server.go`, `fs/ggml/ggml.go`, `llama/llama.go` | `tq25`, `tq35`, aliases, memory accounting | Complete | Request override now sits above env fallback |
| Plan 2 | Request/model override | `api/types.go`, `llm/server.go` | `KVCacheType`, `resolveKVCacheMode` | Complete | Request/model/env precedence finalized |
| Plan 3 | Pure-Go codec | `turboquant/` | presets, rotation, packing, decode, scoring | Complete | `ScoreEncodedVector` reused by fast path |
| Plan 4 | Phase A cache integration | `kvcache/turboquant.go` | packed K/V storage, fallback decode | Complete | Go-managed packed storage retained |
| Plan 5 | CPU compressed-K path | `ml/backend/ggml/ggml.go` | compressed-K attention path | Complete | CPU first, dense fallback retained |
| Productization | `ollama run --turboquant` | `cmd/cmd.go` | CLI flag, preload + generate/chat path | Complete | `off` maps to `f16` |
| Productization | `ollama-bench -turboquant` | `cmd/bench/bench.go` | benchmark request option and labels | Complete | Header reports KV mode and expected path |
| Productization | Modelfile defaults | `docs/modelfile.mdx`, `api/FormatParams` path | `PARAMETER kv_cache_type ...` | Complete | No new instruction added |
| Productization | Public docs | `docs/turboquant.mdx`, `docs/cli.mdx`, `docs/api.md`, `README.md`, `docs/docs.json` | usage + compatibility docs | Complete | GGUF weight quant vs KV quant separated |
| Productization | Final validation layer | `docs/turboquant_completion_scorecard.md`, `scripts/test_turboquant_completion.ps1`, `scripts/score_turboquant_completion.ps1` | final artifact checks | Complete | Complements Plan 1-5 scripts |

## Branch-Specific Adaptations

- The original backend plan assumed `set_rows`-driven GGML cache storage.
- This branch keeps TurboQuant KV storage in Go-managed side storage in `kvcache/turboquant.go`.
- The CPU fast path consumes compressed K rows handed off from `TurboQuantCache.Get(...)`.
- Dense fallback remains the compatibility path when the fast path is unavailable.

