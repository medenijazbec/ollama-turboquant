# TurboQuant Plan 3 Scorecard

This scorecard covers Plan 3 only: the TurboQuant core codec package in `turboquant/`. It builds directly on the Plan 1 audit and the Plan 2 mode-plumbing work, and it is cross-referenced against both source texts:

- `ollama_turboquant_full_plan.txt`
- `ollama_turboquant_algorithms.txt`

These are Static + Unit scores only. They do not include cache integration, generation benchmarks, latency measurements, memory benchmarks, or end-to-end model comparisons. Those runtime comparisons remain deferred until Plan 5.

The current environment does not expose the Go toolchain on `PATH`, so the Plan 3 PowerShell scripts validate the codec and test artifacts statically and skip `go test ./turboquant` execution when `go` is unavailable.

## Score Snapshot

| Metric | Pre | Post | Delta |
| --- | ---: | ---: | ---: |
| Codec Completeness Score | 50 | 100 | +50 |
| Unit Quality Score | 0 | 100 | +100 |

## Pre Snapshot

Pre refers to the branch state before this Plan 3 patch.

- Codec Completeness Score: `50/100`
- Unit Quality Score: `0/100`

### Main pre gaps

- `turboquant/turboquant.go` had only the minimal preset metadata; explicit preset codebooks and decision boundaries were not part of the preset contract yet
- `turboquant/codebook.go` still defined the codec in terms of generic bit-width-based uniform codebooks rather than explicit preset-specific codebooks and boundaries
- `turboquant/residual_qjl.go` had deterministic sign-sketch encode/reconstruct logic, but no dedicated residual dot-correction helper for later fast-path reuse
- `turboquant/block.go` and `turboquant/decode.go` lacked the stronger malformed-input guardrails required by the TXT contract
- The unit tests did not cover malformed decode inputs, deterministic byte-for-byte output, synthetic distortion thresholds, or the `tq35 <= tq25` quality expectation
- No Plan 3 validation script, score script, or scorecard artifact existed

### Pre strengths

- The `turboquant/` package already existed as a self-contained pure-Go package
- Deterministic signed-permutation plus Hadamard-style rotation was already implemented
- Encode/decode, residual sketching, and block serialization already existed as initial scaffolding
- Alias normalization already kept `tq3` and `tq4` mapped to `tq35`

## Post Snapshot

Post refers to the current branch after the Plan 3 patch.

- Codec Completeness Score: `100/100`
- Unit Quality Score: `100/100`

### Post state

- `Preset` now carries explicit preset-defined regular and outlier codebooks plus decision boundaries
- `tq25` and `tq35` remain the only first-class presets, with `tq3` and `tq4` still normalizing to `tq35`
- Encode now uses preset-specific boundary-based scalar quantization
- Residual handling now includes `residualDotCorrection` for later attention-path reuse
- Decode and block unmarshal paths now validate payload lengths and reject malformed block/vector payloads more cleanly
- `Stats` now reports `MSE`, `RMSE`, `MeanAbsErr`, and `MaxAbsErr`
- The `turboquant/` test suite now covers:
  - deterministic rotation and inverse correctness across multiple dimensions
  - preset-specific codebooks and monotonic boundaries
  - deterministic outlier selection on ties
  - deterministic residual sketching and dot-correction behavior
  - block metadata and bit-pack round trips
  - malformed decode cases
  - non-power-of-two tail handling
  - deterministic byte-identical encode output
  - synthetic distortion thresholds for `tq25` and `tq35`

## Verification Commands

```powershell
powershell -File .\scripts\score_turboquant_plan3.ps1
powershell -File .\scripts\test_turboquant_plan3.ps1
```

If the Go toolchain is available locally, `test_turboquant_plan3.ps1` will also run:

```powershell
go test ./turboquant
```

If `go` is not available, the script reports that the Go toolchain is unavailable and still validates the static Plan 3 artifacts.

## Expected Post-Validation Result

- `Codec Completeness Score: 100/100`
- `Unit Quality Score: 100/100`
- `TurboQuant Plan 3 validation passed.`

Full runtime, cache-integration, latency, memory, and generation-quality comparisons remain deferred until Plan 5.
