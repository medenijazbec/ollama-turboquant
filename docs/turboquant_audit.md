# TurboQuant KV Pipeline Audit

## Scope

This document implements Plan 1 only. It is an audit deliverable for the KV-cache pipeline in the new Ollama engine, not a claim that full TurboQuant is already implemented today.

The audit traces `OLLAMA_KV_CACHE_TYPE` through:

- `envconfig/config.go`
- `llm/server.go`
- `fs/ggml/ggml.go`
- `runner/ollamarunner/cache.go`
- `runner/ollamarunner/runner.go`
- `kvcache/cache.go`
- `kvcache/causal.go`
- `ml/backend.go`
- `ml/nn/attention.go`
- `ml/backend/ggml/ggml.go`
- `ml/backend/ggml/ggml/src/ggml-backend.cpp`
- `llama/llama.go`

It also calls out current-tree drift: this branch already contains later-phase TurboQuant work such as `DTypeTQ25`, `DTypeTQ35`, `kvcache/turboquant.go`, and explicit legacy rejection in `llama/llama.go`. Plan 1 still documents the original seams and the remaining handoff points for Plans 2-5.

## Source Basis

This audit is cross-referenced against both source texts in `../../turboquant`:

- `ollama_turboquant_full_plan.txt`
- `ollama_turboquant_algorithms.txt`

Those two files define the implementation contract for this audit:

- `ollama_turboquant_full_plan.txt` defines the required user-visible modes, target files, Phase A versus Phase B boundary, and the instruction that Phase A alone must not be presented as full TurboQuant.
- `ollama_turboquant_algorithms.txt` defines the algorithm stages: deterministic rotation, codebook quantization, residual QJL-style sign sketch, packed KV block storage, fallback decode, fused compressed-attention logits, and cache-copy/resume semantics.

## Cross-Reference Matrix

This cross-reference matrix maps the TXT requirements to the current repo seams and symbols that Plan 1 must hand off to later plans.

| TXT requirement | Source text | Repo file(s) | Function / symbol | Plan handoff |
| --- | --- | --- | --- | --- |
| Reuse `OLLAMA_KV_CACHE_TYPE` | `ollama_turboquant_full_plan.txt` | `envconfig/config.go` | `KvCacheType` | Plan 2 mode plumbing |
| Normalize and validate KV cache type | Both TXT files | `llm/server.go` | `normalizeKVCacheType`, `SupportsKVCacheType`, `KVCacheTypeIsQuantized` | Plan 2 |
| Report supported KV cache types and estimate memory | `ollama_turboquant_full_plan.txt` | `fs/ggml/ggml.go` | `SupportsKVCacheType`, `KVCacheTypeIsQuantized`, `GraphSize`, `kvCacheBytesPerElement` | Plan 2 |
| Map cache mode to backend dtype | `ollama_turboquant_full_plan.txt` | `runner/ollamarunner/cache.go`, `ml/backend.go` | `kvCacheTypeFromStr`, `cache.Init`, `DType` | Plan 2 |
| Construct cache during model startup | `ollama_turboquant_full_plan.txt` | `runner/ollamarunner/runner.go` | `NewInputCache` call from `allocModel` | Plan 4 |
| Preserve cache interface semantics | Both TXT files | `kvcache/cache.go` | `Get`, `Put`, `CopyPrefix`, `CanResume`, `Remove` | Plan 4 |
| Reuse causal metadata behavior | Both TXT files | `kvcache/causal.go` | `StartForward`, `Get`, `Put`, `CopyPrefix`, `CanResume`, `Remove` | Plan 4 |
| Store typed tensors and shape-compatible windows | Both TXT files | `kvcache/causal.go`, `ml/nn/attention.go` | cache window shapes, `cache.Put`, `cache.Get` | Plan 4 Phase A fallback |
| Keep attention entrypoint stable | Both TXT files | `ml/nn/attention.go`, `ml/backend.go`, `ml/backend/ggml/ggml.go` | `ScaledDotProductAttention` | Plan 4 and Plan 5 |
| Add backend dtype / compressed tensor support | `ollama_turboquant_full_plan.txt` | `ml/backend.go`, `ml/backend/ggml/ggml.go` | `DType`, `Tensor.DType`, backend dtype conversion | Plan 5 |
| Add backend upload / scheduling seam for packed blocks | `ollama_turboquant_algorithms.txt` | `ml/backend/ggml/ggml/src/ggml-backend.cpp` | `ggml_backend_tensor_set_async`, `ggml_backend_graph_compute_async`, scheduler reservation code | Plan 5 |
| Add CPU fast-path files | Both TXT files | `ml/backend/ggml/ggml/src/ggml.c`, `ml/backend/ggml/ggml/src/ggml-quants.c` | GGML type definitions, block codecs, vec-dot and attention logic | Plan 5 |
| Keep legacy runner explicit | `ollama_turboquant_full_plan.txt` | `llama/llama.go` | `NewContextParams`, legacy `kvCacheTypeFromStr` | Plan 2 |

## OLLAMA_KV_CACHE_TYPE Trace

### 1. Environment entry

- `envconfig/config.go` exports `KvCacheType = String("OLLAMA_KV_CACHE_TYPE")`.
- That is the sole environment entrypoint for KV-cache mode selection in the new engine.

### 2. Load-time normalization and validation

- `llm/server.go` reads `envconfig.KvCacheType()`.
- `llm/server.go` normalizes the mode with `normalizeKVCacheType`.
- `llm/server.go` validates the effective value with `SupportsKVCacheType`.
- `llm/server.go` gates quantized modes with `KVCacheTypeIsQuantized` plus flash attention checks.
- `llm/server.go` threads the normalized result into `LoadRequest.KvCacheType`.

### 3. Memory planning and model fit calculation

- `llm/server.go` passes `LoadRequest.KvCacheType` to `s.ggml.GraphSize(...)`.
- `fs/ggml/ggml.go` implements `GraphSize(context, batch, numParallel, kvCacheType, useFlashAttention)`.
- `GraphSize` calls `kvCacheBytesPerElement(kvCacheType)` to estimate KV storage cost for non-recurrent attention layers.
- `GraphSize` also uses `kvCacheBytesPerElement("f32")` for recurrent-path accounting.
- `fs/ggml/ggml.go` therefore remains the model-fit seam where any new TurboQuant storage ratio must be reflected.

### 4. Runner cache construction

- `runner/ollamarunner/runner.go` calls `NewInputCache(...)` during `allocModel`.
- `runner/ollamarunner/cache.go` converts the normalized string mode via `kvCacheTypeFromStr`.
- `runner/ollamarunner/cache.go` then calls `cache.Init(model.Backend(), dtype, numSlots, int(numCtx), batchSize)`.
- This is the seam where Plan 4 chooses between the causal cache path and `TurboQuantCache`.

### 5. Cache interface boundary

- `kvcache/cache.go` defines the stable runtime contract:
  - `Init`
  - `StartForward`
  - `Get`
  - `Put`
  - `CopyPrefix`
  - `CanResume`
  - `Remove`
- Both TXT files assume TurboQuant integrates here first, because this interface lets Phase A decode back to ordinary tensors without changing model code.

### 6. Causal metadata and tensor windowing

- `kvcache/causal.go` owns the decoder-style KV metadata, active history window, mask generation, and slot removal behavior.
- `Put` writes the incoming tensors into layer-local cache tensors.
- `Get` builds a history window and returns `key`, `value`, and `mask` for the current batch.
- `CopyPrefix`, `CanResume`, and `Remove` define the copy-prefix and resume semantics that the algorithm TXT explicitly calls out as mandatory for packed TurboQuant blocks.

### 7. Attention entrypoint

- `ml/nn/attention.go` writes the latest K/V through `cache.Put(ctx, key, value)`.
- It then immediately re-reads history with `key, value, mask := cache.Get(ctx)`.
- That file forwards the tensors into `ScaledDotProductAttention`.
- This is why Phase A can remain cache-local: `ml/nn/attention.go` only requires that `Get` returns shape-compatible tensors and a mask.

### 8. Backend execution

- `ml/backend.go` defines the abstract `ScaledDotProductAttention` interface plus `DType`, `CacheConfig`, `PermutedV`, and `MaskDType`.
- `ml/backend/ggml/ggml.go` implements backend dtype conversion, cache config behavior, and GGML attention dispatch.
- `ml/backend/ggml/ggml/src/ggml-backend.cpp` is not the cache-mode parser, but it is the scheduler and host/device tensor movement seam that Plan 5 must use for packed block upload, reservation, and compute dispatch through `ggml_backend_tensor_set_async` and `ggml_backend_graph_compute_async`.

## Hard-Coded Cache-Type Assumptions

This section records every material hard-coded cache-type assumption that either started from `f16` / `q8_0` / `q4_0` only, or still constrains how TurboQuant can land cleanly.

### Config and validation assumptions

- `fs/ggml/ggml.go` is the capability layer through `SupportsKVCacheType` and `KVCacheTypeIsQuantized`.
- `llm/server.go` treats quantized cache types as flash-attention dependent. That is still a hard-coded behavior gate, even if the accepted set has expanded on this branch.
- `normalizeKVCacheType` is the normalization seam for compatibility aliases such as `tq3` and `tq4`.

### DType assumptions

- `ml/backend.go` is where the `DType` enum constrains which cache storage modes the backend can represent.
- `runner/ollamarunner/cache.go` is where string cache types become `ml.DType` via `kvCacheTypeFromStr`.
- `ml/backend/ggml/ggml.go` is where GGML dtype conversion constrains what the backend can allocate and return.

### Legacy assumptions

- `llama/llama.go` still has legacy `kvCacheTypeFromStr` logic that only maps known llama.cpp GGML types.
- The old baseline behavior was silent `f16` fallback for unknown values.
- Current-tree drift already adds explicit reject logic in `NewContextParams` for TurboQuant modes, which is the correct seam to preserve.

### Backend fast-path assumptions

- `ml/backend/ggml/ggml.go` and `ml/backend/ggml/ggml/src/ggml-backend.cpp` still assume tensor storage is expressed in backend-supported dtypes or host buffers that the scheduler understands.
- `ml/backend/ggml/ggml/src/ggml.c` and `ml/backend/ggml/ggml/src/ggml-quants.c` are the eventual hard-coded sites for new packed `GGML_TYPE_TQ25` / `GGML_TYPE_TQ35` block definitions and dot-product kernels.

## KV Memory Estimation

The current KV memory-estimation callsites are:

1. `fs/ggml/ggml.go` -> `GraphSize(...)`
   - Uses `kvCacheBytesPerElement(kvCacheType)` for attention-layer KV cost.
   - Uses `kvCacheBytesPerElement("f32")` for recurrent-path accounting.
2. `llm/server.go` -> `s.ggml.GraphSize(...)`
   - Consumes the per-layer memory plan to reserve model execution memory and decide fit/offload behavior.

This is the exact place where Plan 2 must keep memory accounting honest for `tq25` and `tq35`.

## Cache Contract And Tensor Shapes

### Cache contract

`kvcache/cache.go` requires every cache implementation to preserve:

- `Init(backend, dtype, maxSequences, capacity, maxBatch)`
- `StartForward(ctx, batch, reserve)`
- `Get(ctx) (key, value, mask)`
- `Put(ctx, key, value)`
- `CopyPrefix(srcSeq, dstSeq, len)`
- `CanResume(seq, pos)`
- `Remove(seq, beginIndex, endIndex)`

### Causal cache tensor shapes

From `kvcache/causal.go`:

- Incoming `key` shape: `[head_dim, kv_heads, batch]`
- Incoming `value` shape: `[head_dim, kv_heads, batch]`
- Stored K tensor shape: `[head_dim, kv_heads, cache_cells]`
- Stored V tensor shape:
  - `[cache_cells, head_dim, kv_heads]` when `PermutedV` is enabled
  - `[head_dim, kv_heads, cache_cells]` otherwise
- `Get` returns a history window and a mask with shape `[history, batch]`

### Semantic requirements from the algorithm TXT

- `Put` must preserve token append order per sequence and per layer.
- `Get` must return only the active history window required by the current batch.
- `CopyPrefix` must copy packed or unpacked KV state without changing semantic positions.
- `CanResume` must remain sensitive to preset/version compatibility once TurboQuant side metadata exists.
- `Remove` must continue to support partial truncation and full eviction.

## Attention And Backend Insertion Points

### Phase A

Phase A is correctness-first fallback compute. It is not full TurboQuant.

- Best insertion point: `kvcache/cache.go` interface with a `TurboQuantCache`.
- `ml/nn/attention.go` remains unchanged except for consuming ordinary tensors returned by `Get`.
- The fallback path decodes only the active KV window, then routes through the existing `ScaledDotProductAttention` path in `ml/backend.go` and `ml/backend/ggml/ggml.go`.

### Phase B

Phase B is the compressed-attention fast path required for a full TurboQuant claim.

- `ml/backend.go` must expose or recognize backend capability for TurboQuant-compressed attention.
- `ml/backend/ggml/ggml.go` must recognize TurboQuant dtypes or compressed-cache tensors in the attention path.
- `ml/backend/ggml/ggml/src/ggml-backend.cpp` is the scheduler and upload seam for packed block movement, reservation, and graph execution through `ggml_backend_tensor_set_async` and `ggml_backend_graph_compute_async`.
- `ml/backend/ggml/ggml/src/ggml.c` must define any new `GGML_TYPE_TQ25` / `GGML_TYPE_TQ35` tensor types and their behavior.
- `ml/backend/ggml/ggml/src/ggml-quants.c` must define block layouts, quant/dequant helpers, and vec-dot support for compressed K blocks.

## Legacy Runner Handling

- `runner/llamarunner/runner.go` remains the legacy llama.cpp path.
- `llama/llama.go` owns `NewContextParams`, which is the explicit compatibility seam for legacy KV cache typing.
- The old baseline seam silently fell back to `GGML_TYPE_F16` for unknown cache types.
- Current-tree drift already improves this by rejecting `tq25` and `tq35` in `NewContextParams`.
- Plan 1 requirement: keep that explicit reject behavior. Do not silently pretend the legacy runner shipped true TurboQuant.

## Plan 2-5 Insertion Points

### Plan 2

- `envconfig/config.go`
- `llm/server.go`
- `fs/ggml/ggml.go`
- `runner/ollamarunner/cache.go`
- `ml/backend.go`
- `llama/llama.go`

Work: mode plumbing, alias normalization, logging, validation, memory estimation, and legacy reject behavior.

### Plan 3

- New `turboquant/` package
- `rotation.go`
- `codebook.go`
- `residual_qjl.go`
- `block.go`
- `encode.go`
- `decode.go`
- Tests for determinism, round-trip, block layout, and residual correction

Work: implement the algorithm core from `ollama_turboquant_algorithms.txt`.

### Plan 4

- `kvcache/cache.go`
- `kvcache/causal.go`
- `kvcache/turboquant.go`
- `runner/ollamarunner/cache.go`
- `runner/ollamarunner/runner.go`
- `ml/nn/attention.go`

Work: integrate `TurboQuantCache`, preserve `Get` / `Put` semantics, and use Phase A fallback decode.

### Plan 5

- `ml/backend.go`
- `ml/backend/ggml/ggml.go`
- `ml/backend/ggml/ggml/src/ggml-backend.cpp`
- `ml/backend/ggml/ggml/src/ggml.c`
- `ml/backend/ggml/ggml/src/ggml-quants.c`

Work: add compressed attention support, block types, upload paths, and CPU fast-path validation.

## Gap Analysis

Plan 1 covers:

- repo seam discovery
- source-text cross-reference
- KV pipeline tracing
- hard-coded assumption inventory
- memory-estimation callsite inventory
- cache contract and tensor shape inventory
- legacy runner seam documentation
- exact insertion points for Plans 2-5

Plan 1 does not cover:

- new runtime behavior
- new `ml.DType` work
- TurboQuant codec implementation
- TurboQuant cache packing
- generation correctness testing
- fused compressed attention
- benchmark or quality claims

The current branch already contains some later-phase drift, but this audit still separates the implementation boundary correctly:

- Phase A means fallback decode and existing attention compute.
- Phase B means compressed K attention with backend support.
- Full TurboQuant should only be claimed once Phase B exists and is validated.
