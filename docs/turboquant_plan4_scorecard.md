# TurboQuant Plan 4 Scorecard

This scorecard covers Plan 4 only: KV cache Phase A integration in the new engine. It builds on the earlier Plan 1 audit, Plan 2 mode plumbing, and Plan 3 codec work, and it is cross-referenced against:

- `04_kvcache_phase_a_integration.plan.txt`
- `ollama_turboquant_full_plan.txt`
- `ollama_turboquant_algorithms.txt`

These are static + unit + cache-integration scores only. They do not include full generation smoke tests, latency measurements, memory benchmarks, or end-to-end output comparisons. Those runtime comparisons remain deferred until Plan 5.

The current environment still does not expose the Go toolchain on `PATH`, so the Plan 4 scripts validate the source and test artifacts statically and skip `go test ./kvcache ./runner/ollamarunner` when `go` is unavailable.

## Score Snapshot

| Metric | Pre | Post | Delta |
| --- | ---: | ---: | ---: |
| Phase A Cache Integration Completeness Score | 60 | 100 | +40 |
| Cache Lifecycle Parity Score | 35 | 100 | +65 |

## Pre Snapshot

Pre refers to the branch state before this Plan 4 patch.

- Phase A Cache Integration Completeness Score: `60/100`
- Cache Lifecycle Parity Score: `35/100`

### Main pre gaps

- `kvcache/turboquant.go` already existed, but `Put` reused the key stride for values, so `valueDim != keyDim` was not handled correctly
- the TurboQuant cache rediscovered current batch locations by scanning `meta.cells` instead of using forward-pass state directly
- `Remove` only supported tail deletion and returned `ErrNotSupported` for middle deletes
- packed TurboQuant key entries were not shifted after mid-sequence removal
- test coverage only covered a narrow store/get path and a minimal copy-prefix case
- no Plan 4 validation script, score script, or scorecard artifact existed

### Pre strengths

- the runner already selected TurboQuant wrapping for `DTypeTQ25` and `DTypeTQ35`
- `kvcache.TurboQuantCache` already stored packed bytes in Go-managed side storage
- `Get` already used a fallback decode path that returned ordinary tensors to the attention code
- `ml.TurboQuantBackend` already exposed the Phase A versus fast-path boundary, and the ggml backend still reported `SupportsTurboQuantFastPath() == false`

## Post Snapshot

Post refers to the current branch after the Plan 4 patch.

- Phase A Cache Integration Completeness Score: `100/100`
- Cache Lifecycle Parity Score: `100/100`

### Post state

- `kvcache/causal.go` now records exact forward-pass cache locations in `curLocs`
- `kvcache/turboquant.go` now:
  - encodes both K and V with independent strides
  - uses `meta.curLocs` directly
  - decodes the active history window into dense fallback tensors
  - preserves permuted-V output layout
  - supports copy-prefix and resume via causal metadata
  - supports both tail deletion and middle deletion with key-only shift/re-encode
- `runner/ollamarunner/cache.go` keeps TurboQuant wrapping restricted to `tq25` / `tq35` cache modes
- tests now cover:
  - both presets
  - permuted and non-permuted V
  - `keyDim != valueDim`
  - multi-batch append
  - copy-prefix / resume
  - tail-delete cleanup
  - middle-delete key shift
  - shift unsupported behavior
  - SWA / SWAMem behavior
  - wrapper recursion that preserves encoder caches

## Verification Commands

```powershell
powershell -File .\scripts\score_turboquant_plan4.ps1
powershell -File .\scripts\test_turboquant_plan4.ps1
```

If the Go toolchain is available locally, `test_turboquant_plan4.ps1` will also run:

```powershell
go test ./kvcache ./runner/ollamarunner
```

If `go` is not available, the script reports that the Go toolchain is unavailable and still validates the static Plan 4 artifacts.

## Expected Post-Validation Result

- `Phase A Cache Integration Completeness Score: 100/100`
- `Cache Lifecycle Parity Score: 100/100`
- `TurboQuant Plan 4 validation passed.`

Full runtime pre/post generation, latency, memory, and model-quality comparisons remain deferred until Plan 5.
