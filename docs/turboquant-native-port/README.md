# TurboQuant Native Port Checkpoints

This directory tracks checkpoint-scoped work for the native-port lane.

Available in this checkout:

- [08-large-context-benchmark-plan.md](08-large-context-benchmark-plan.md)
- [08-test-matrix.md](08-test-matrix.md)
- [09-agentic-and-recall-benchmarks.md](09-agentic-and-recall-benchmarks.md)
- [10-large-context-memory-benchmarks.md](10-large-context-memory-benchmarks.md)

Notes:

- Checkpoints 01-07 were referenced by earlier planning notes but are not present in this checkout.
- Checkpoint 08 extends the existing `cmd/benchkv` harness instead of introducing a separate large-context benchmark binary.
- The native A/B layout is baseline on `11438` and TurboQuant on `11439`.
- Stable default lane: MSE/Lloyd-Max style TurboQuant with QJL off by default.
- Experimental: K-side QJL, segmented `576 -> 256+256+64`, residual FP16 tail, and conservative surface-aware policy handling.
- Blocked: V-side QJL requests.
- Scaffold-only: experimental weight quantization metadata, with no active model-weight compression path yet.
