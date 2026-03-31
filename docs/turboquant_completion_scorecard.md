# TurboQuant Completion Scorecard

This scorecard covers the final TurboQuant productization pass:

- final completion audit against Plans 1-5
- per-request `kv_cache_type` override
- `ollama run --turboquant`
- `ollama-bench -turboquant`
- public docs and benchmark placeholders

It builds on:

- `docs/turboquant_audit.md`
- `docs/turboquant_plan2_scorecard.md`
- `docs/turboquant_plan3_scorecard.md`
- `docs/turboquant_plan4_scorecard.md`
- `docs/turboquant_plan5_scorecard.md`

## Scope

This scorecard is for:

- implementation completeness
- product-surface completeness

Runtime benchmark numbers are not hardcoded here. They are produced by the benchmark workflow and copied into the TurboQuant docs table separately.

## Pre / Post

- Implementation Completeness Score: `85 -> 100`
- Product Surface Completeness Score: `40 -> 100`

## What Changed In This Pass

- Added `api.Runner.KVCacheType`
- Added request/model/env precedence resolution in `llm/server.go`
- Added `ollama run --turboquant`
- Added `ollama-bench -turboquant`
- Added Modelfile docs for `PARAMETER kv_cache_type`
- Added public TurboQuant docs page and navigation entry
- Added final completion matrix and validation scripts

## Validation Commands

```powershell
powershell -File .\scripts\score_turboquant_completion.ps1
powershell -File .\scripts\test_turboquant_completion.ps1
```

If the Go toolchain is available locally, the completion test script also runs:

```powershell
go test ./api ./cmd ./server ./llm ./kvcache ./ml/backend/ggml ./turboquant
```

