# TurboQuant Paper Path

This note is the commit-facing summary for the current TurboQuant implementation and the latest benchmark readout.

## Summary

This fork now treats `tq25` and `tq35` as the default paper-style TurboQuant path:

- deterministic random orthogonal rotation
- scalar Lloyd-Max quantization in rotated space
- key-side product scoring with a 1-bit QJL-style residual sketch
- value-side MSE quantization without residual correction
- runtime metadata reporting `kv_algo_resolved=paper`

PolarQuant is not implemented in this phase.

## Implemented In

### Algorithm core

- [turboquant/turboquant.go](../turboquant/turboquant.go)
- [turboquant/rotation.go](../turboquant/rotation.go)
- [turboquant/codebook.go](../turboquant/codebook.go)
- [turboquant/encode.go](../turboquant/encode.go)
- [turboquant/decode.go](../turboquant/decode.go)
- [turboquant/residual_qjl.go](../turboquant/residual_qjl.go)
- [turboquant/block.go](../turboquant/block.go)

### KV cache integration

- [kvcache/turboquant.go](../kvcache/turboquant.go)
- [runner/ollamarunner/cache.go](../runner/ollamarunner/cache.go)

### Runtime and API metadata

- [runner/ollamarunner/runner.go](../runner/ollamarunner/runner.go)
- [api/types.go](../api/types.go)
- [server/routes.go](../server/routes.go)

### Benchmark validation and telemetry

- [cmd/benchkv/runner.go](../cmd/benchkv/runner.go)
- [cmd/benchkv/output.go](../cmd/benchkv/output.go)
- [cmd/benchkv/types.go](../cmd/benchkv/types.go)
- [cmd/benchkv/monitor_gpu.go](../cmd/benchkv/monitor_gpu.go)
- [docker-compose.ollama-bench.yml](../docker-compose.ollama-bench.yml)

## Preset Mapping

### `tq25`

- keys: 2-bit primary scalar quantizer
- keys: 1-bit QJL residual sketch
- values: 2-bit scalar quantizer

### `tq35`

- keys: 3-bit primary scalar quantizer
- keys: 1-bit QJL residual sketch
- values: 3-bit scalar quantizer

The preset seeds are fixed in [turboquant/turboquant.go](../turboquant/turboquant.go), so rotation and residual sketch generation are deterministic for a given dimension and preset.

## Implemented Data Path

### Rotation

[turboquant/rotation.go](../turboquant/rotation.go) builds a dense orthogonal matrix from a seeded Gaussian matrix and caches it by `(dim, seed)`.

Implementation details:

- rows are sampled from a Gaussian distribution
- Gram-Schmidt orthogonalization is applied
- row signs are normalized deterministically
- the matrix is cached and reused across tokens, heads, and layers

### Scalar codebooks

[turboquant/codebook.go](../turboquant/codebook.go) builds deterministic Lloyd-Max scalar codebooks for the selected bit width.

Implementation details:

- codebooks are cached by `(dim, bits)`
- initialization is deterministic
- scalar boundaries are derived from centroid midpoints
- this replaces the earlier outlier-based path

### Key path

[turboquant/encode.go](../turboquant/encode.go) routes keys through the product objective:

1. rotate the key vector
2. quantize rotated coordinates with the primary scalar codebook
3. reconstruct the primary approximation in rotated space
4. compute the residual
5. sketch the residual with a 1-bit Gaussian-sign projection in [turboquant/residual_qjl.go](../turboquant/residual_qjl.go)
6. pack primary codes plus residual sketch into a versioned block

At attention time, [turboquant/decode.go](../turboquant/decode.go) scores keys using:

- the base dot product from the primary scalar reconstruction
- plus `residualDotCorrection(...)` from the residual sketch

### Value path

Values use the MSE path only:

1. rotate the value vector
2. quantize rotated coordinates with the value codebook
3. pack only the primary scalar codes
4. decode by dequantizing primary codes and applying the inverse rotation

No residual sketch is stored for values in this phase.

### Packed format

[turboquant/block.go](../turboquant/block.go) stores a self-describing versioned block:

- `Version`
- `PresetID`
- `Role`
- `Objective`
- original and padded dimensions
- scalar bit width
- rotation seed
- codebook id
- QJL row count
- aux layout id
- primary packed indices
- residual sketch payload

Keys use:

- `role=key`
- `objective=product`

Values use:

- `role=value`
- `objective=mse`

## Pseudocode

### Setup

```text
setup(dim, preset):
  R = BuildRotation(dim, preset.RotationSeed)
  key_codebook = scalarCodebook(dim, preset.KeyPrimaryBits)
  value_codebook = scalarCodebook(dim, preset.ValueBits)
  qjl_rows = preset.KeyQJLRows(dim)
  return {R, key_codebook, value_codebook, qjl_rows}
```

### Key encode

```text
encode_key(x, preset):
  state = setup(len(x), preset)
  y = ApplyRotation(x, state.R)
  primary_codes, y_hat = scalar_quantize(y, state.key_codebook)
  residual = y - y_hat
  sketch = encodeResidual(y, y_hat, state.qjl_rows, preset_seed_mix)
  return pack_v2(role=key, objective=product,
                 primary_codes, sketch, metadata)
```

### Value encode

```text
encode_value(x, preset):
  state = setup(len(x), preset)
  y = ApplyRotation(x, state.R)
  primary_codes = scalar_quantize(y, state.value_codebook)
  return pack_v2(role=value, objective=mse,
                 primary_codes, metadata)
```

### Key score

```text
score_key(q, packed_key):
  state = setup(packed_key.dim, packed_key.preset)
  q_rot = ApplyRotation(q, state.R)
  y_hat = decode_primary_codes(packed_key.primary_codes, state.key_codebook)
  base = dot(q_rot, y_hat)
  corr = residualDotCorrection(q_rot, packed_key.sketch)
  return base + corr
```

### Value decode

```text
decode_value(packed_value):
  state = setup(packed_value.dim, packed_value.preset)
  y_hat = decode_primary_codes(packed_value.primary_codes, state.value_codebook)
  return ApplyInverseRotation(y_hat, state.R)
```

## Fidelity vs Engineering Glue

### Paper-faithful structure

- random orthogonal rotation
- scalar quantization in rotated space
- key-side product estimator with a 1-bit residual sketch
- MSE reconstruction path for values

### Engineering glue in this fork

- deterministic seeds
- cached rotations and codebooks
- one packed block format for runtime use
- KV cache wrapper integration
- runtime metadata validation in the benchmark harness

## Benchmark Report

The benchmark comparison below uses:

- **Test 1**: `baseline f16` vs `turbo runtime f16`
- **Tests 2-7**: `baseline f16` vs `turbo tq35`

Headline percentages use end-to-end `total_ms` unless noted otherwise.

## Public Test Server

The benchmark server used for the public numbers in this report had:

- 2x NVIDIA Tesla M40 24 GB GPUs
- CUDA driver 12.2
- about 94 GiB system RAM
- about 80 GiB swap available

These are modest test resources for long-context KV-cache work, especially because Tesla M40 cards are passively cooled and can be sensitive to airflow, thermal conditions, and long sustained runs.

## Benchmark Scope Note

The published runs in this report are intentionally short-form benchmark passes.

Reason:

- available hardware is limited
- the M40 platform is older and thermally constrained
- full exhaustive long-context sweeps would take much longer and can be distorted by thermal instability

Because of that, these numbers should be read as:

- a public first-pass benchmark
- useful directional evidence
- not the final word on TurboQuant performance on better hardware

Anyone is encouraged to:

- run the benchmark on their own hardware
- publish full results
- compare against these numbers
- report stronger or weaker outcomes

### 1. Regression gate

This is not a TurboQuant-on test. It is only fork-vs-stock in `f16`.

- Headline: **42.2% faster overall**
- Decode: **+77.8%**
- TTFT: **0.9% better**
- Prefill: about **+1.0%**

Note: this result used the median because the file contained one obvious outlier row per host.

### 2. Same-runtime paper TurboQuant proof

Matched row: `prefill-heavy`, `ctx=16384`, `conc=1`

- Headline: **6.6% worse overall**
- Decode: **+190.0%**
- Prefill: **7.0% worse**
- TTFT: **22.4% worse**
- Capacity: **no gain shown**

Interpretation: decode improved sharply, but the overall result was still negative.

### 3. Capacity proof for paper TurboQuant

Matched rows: `ctx=16384` and `32768`, `conc=1`

- Headline: **19.8% faster overall**
- Decode: **+328.5%**
- Prefill: **+14.8%**
- TTFT: **10.8% better**
- Capacity: **0% proven gain**

Observed capacity note:

- both sides reached the same `max_stable_ctx_x_conc=32768`
- matched rows had the same `live_kv_tokens_total`

### 4. Spill-allowed survival

Matched rows: `ctx=32768`, `conc=1` and `2`

- Headline: **54.8% faster overall**
- Decode: **+453.7%**
- Prefill: **+90.6%**
- TTFT: **52.0% better**
- Capacity: **0% proven gain**

Observed capacity note:

- both sides reached the same `max_stable_ctx_x_conc=65536`
- matched rows had the same live KV totals

This is the strongest speed result in the current set.

### 5. Fast first pass

Matched rows: 4 cells

- Headline: **3.7% faster overall**
- Decode: **+70.7%**
- Prefill: **4.7% worse**
- TTFT: **24.3% worse**

Interpretation: decode improved meaningfully, but TTFT and prefill dragged the overall gain down.

### 6. Short manual same-runtime paper check

Matched rows: `decode-growth`, `ctx=8192` and `16384`

- Headline: **5.9% faster overall**
- Decode: **+42.0%**
- Prefill: **+0.0%**
- TTFT: **37.2% worse**

Interpretation: modest end-to-end gain, but TTFT still regressed.

### 7. Product-outcome check

Matched rows: 1 cell

- Headline: **0.9% faster overall**
- Decode: **+225.0%**
- Prefill: **1.3% worse**
- TTFT: **14.0% worse**
- Capacity: **no gain shown**

Note: the command file later said `--num-ctx 32768`, but the saved CSV row is `16384`, so this test has a command/results mismatch in the current archive.

## Overall Result

### Weighted overall

Across all directly comparable TurboQuant-vs-baseline cells in Tests 2-7, weighted by baseline workload size:

- **TurboQuant `tq35` is 30.2% faster overall**
- Prefill: **+7.9%**
- Decode: **+117.0%**
- TTFT: **23.4% better**

### Conservative unweighted roll-up

If the six TurboQuant test groups are averaged equally:

- **TurboQuant `tq35` is 13.1% better overall**

This is the safer headline when long 32k spill-adjacent cells should not dominate the aggregate.

### Whole turbo stack

If Test 1 is included as part of the whole fork package rather than pure TurboQuant:

- the **full turbo stack is 36.5% faster overall**

This is not a pure TurboQuant number because Test 1 is `turbo f16` vs `baseline f16`.

## Interpretation

The current pattern is:

- TurboQuant helps most in heavier long-context and spill-adjacent workloads
- decode throughput often improves substantially
- TTFT is not consistently better
- prefill is mixed
- end-to-end gains range from slightly negative to very strong depending on workload

## Caveats

These results do **not** prove a memory-capacity win yet.

Why:

- `peak_vram_bytes` was missing in the exported results
- processor-state and residency data were not strong enough for a clean offload proof
- matched baseline vs turbo rows show the same `live_kv_tokens_total`
- the exported host-memory data suggested turbo `tq35` used about **12.5% more peak host RAM overall** across matched Tests 2-7

So the honest read is:

- **Speed:** yes, a real overall win is present
- **Memory/capacity:** not proven by this archive

## Bottom Line

If one number is needed:

- **TurboQuant `tq35` is 30.2% faster overall vs baseline**, weighted by end-to-end runtime across directly comparable cells in Tests 2-7

If a more conservative headline is needed:

- **TurboQuant `tq35` is 13.1% better overall**, averaging the six TurboQuant test groups equally

If the whole fork package is summarized rather than pure TurboQuant:

- the **full turbo stack is 36.5% faster overall**

## Benchmark Notes

Use the command sheet in [kvstress-test-commands.txt](./kvstress-test-commands.txt).

GPU monitoring note:

- `peak_vram_bytes` depends on the bench container having GPU utility access
- [docker-compose.ollama-bench.yml](../docker-compose.ollama-bench.yml) now mounts both results folders and enables GPU utility visibility for [cmd/benchkv/monitor_gpu.go](../cmd/benchkv/monitor_gpu.go)
