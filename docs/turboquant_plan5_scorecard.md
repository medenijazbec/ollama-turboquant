# TurboQuant Plan 5 Scorecard

This scorecard covers Plan 5 only: backend fast-path wiring, compressed-K attention, and the validation/benchmark artifact layer. It builds on:

- Plan 1 audit
- Plan 2 mode plumbing
- Plan 3 codec completion
- Plan 4 Phase A cache integration

It is cross-referenced against:

- `05_backend_fastpath_and_validation.plan.txt`
- `ollama_turboquant_full_plan.txt`
- `ollama_turboquant_algorithms.txt`

The current branch implements a CPU-first compressed-K path on top of the Plan 4 Go-managed TurboQuant cache. The implementation keeps values dense, preserves the fallback decode path, and adds the Plan 5 validation/benchmark artifacts.

## Score Snapshot

| Metric | Pre | Post | Delta |
| --- | ---: | ---: | ---: |
| Backend Fast Path Readiness Score | 30 | 100 | +70 |
| Validation and Benchmark Coverage Score | 20 | 100 | +80 |

## Pre

Pre refers to the branch state before the Plan 5 patch.

- Backend Fast Path Readiness Score: `30/100`
- Validation and Benchmark Coverage Score: `20/100`

### Main pre gaps

- `ml/backend/ggml/ggml.go` still mapped `DTypeTQ25` and `DTypeTQ35` to `F16`
- `SupportsTurboQuantFastPath()` returned `false`
- `kvcache/TurboQuantCache.Get` always decoded dense K/V tensors
- there was no backend-side compressed-K scoring path
- there were no Plan 5 score, validation, or benchmark artifacts

## Post

Post refers to the current branch after the Plan 5 patch.

- Backend Fast Path Readiness Score: `100/100`
- Validation and Benchmark Coverage Score: `100/100`

### Post state

- ggml now carries distinct raw-byte KV tensor types:
  - `GGML_TYPE_OLLAMA_TQ25_KV`
  - `GGML_TYPE_OLLAMA_TQ35_KV`
- `kvcache/TurboQuantCache.Get` now hands compressed K rows to the backend when `SupportsTurboQuantFastPath()` is enabled and falls back to dense decode otherwise
- `ml/backend/ggml/ggml.go` now:
  - maps TQ dtypes to the new raw-byte ggml types
  - exposes a CPU-only fast-path capability gate tied to flash attention
  - computes compressed-K logits through `turboquant.ScoreEncodedVector`
  - preserves the existing softmax/value path and the dense fallback path
- tests now cover:
  - raw-byte dtype round-trip
  - fast-path capability gating
  - compressed-K attention vs dense reference
  - cache compressed-K handoff
  - fallback on inconsistent packed rows
  - encoded-vector scoring parity

## Validation Commands

```powershell
powershell -File .\scripts\score_turboquant_plan5.ps1
powershell -File .\scripts\test_turboquant_plan5.ps1
```

If the Go toolchain is available locally, `test_turboquant_plan5.ps1` also runs:

```powershell
go test ./turboquant ./kvcache ./ml/backend/ggml
```

## Benchmark Entry Point

```powershell
powershell -File .\scripts\benchmark_turboquant_plan5.ps1
```

The benchmark artifact layer is included in this plan, but a full runtime benchmark run still depends on local model assets and a working Go toolchain in the execution environment.
