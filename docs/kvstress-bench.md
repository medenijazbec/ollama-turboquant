# KV Stress Bench

`kvstress-bench` is the live-KV occupancy and capacity benchmark for this fork. It is separate from `ollama-bench`.

- `ollama-bench` remains the short-form latency sanity tool.
- `kvstress-bench` is the primary proof tool for long-context, concurrency, spill onset, and stable operating envelope under KV pressure.

## What It Measures

- peak VRAM and host RAM under different `kv_cache_type` modes
- prefill throughput under large prompt/context pressure
- decode throughput as KV grows during generation
- live KV occupancy per request and per concurrent epoch
- full-GPU vs mixed CPU/GPU residency
- max stable context and concurrency before spill or failure
- actual runtime KV path as reported by the server
- actual runtime TurboQuant algorithm resolution (`kv_algo_resolved`)
- requested vs effective QJL flags
- residual-tail A/B rows on long-context profiles
- sparse KLD vs baseline when score-route logprobs overlap sufficiently
- NIAH retrieval pass/fail and corruption-class reporting

`--timeout 0` means unlimited runtime for preflight, prompt calibration, and benchmark epochs.

## A/B Hosts

- stock Ollama: `http://127.0.0.1:11438`
- TurboQuant Ollama: `http://127.0.0.1:11439`

## Baseline Validity Check

Before using baseline-vs-fork claims, verify the stock host at `11438` is actually running on GPU.

If the stock host logs show:
- `ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected`
- `offloaded 0/... layers to GPU`
- `model weights device=CPU`

then the baseline host is CPU-only and these two comparisons are not valid:
- regression: `stock f16` vs `turbo f16`
- product outcome: `stock f16` vs `turbo tq35`

In that case, use only:
- same-runtime TurboQuant comparisons on the turbo host
- capacity and spill summaries on the turbo host

Recommended validation command:

```bash
kvstress-bench \
  --hosts http://127.0.0.1:11438 \
  --host-labels baseline \
  --model qwen3-30b:instruct2507-udq4kxl \
  --kv-modes f16 \
  --workloads decode-growth \
  --num-ctx 4096 \
  --concurrency 1 \
  --warmup 0 \
  --epochs 1 \
  --output /results/native_baseline_gpu_check_paper.csv \
  --jsonl-output /results/native_baseline_gpu_check_paper.jsonl \
  --summary-output /results/native_baseline_gpu_check_paper.md
```

The baseline host must show:
- `status=ok`
- `full_gpu_residency=true`
- non-empty `processor_state_after`

If it does not, treat the native baseline as an environment problem rather than an algorithm result.

## Example

```bash
kvstress-bench \
  --hosts http://127.0.0.1:11438,http://127.0.0.1:11439 \
  --host-labels baseline,turbo \
  --host-kv-support legacy,request \
  --model qwen3-30b:instruct2507-udq4kxl \
  --kv-modes f16,q8_0,q4_0,tq25,tq35 \
  --profile turbo-benefit \
  --output /results/qwen30b_kvstress.csv \
  --jsonl-output /results/qwen30b_kvstress.jsonl \
  --summary-output /results/qwen30b_kvstress.md
```

## Profiles

- `regression`: stock `f16` vs turbo `f16` sanity gate at `ctx=8192`, `conc=1`, `decode-growth`
- `turbo-benefit`: turbo host only, `f16/tq25/tq35`, full-GPU speed grid and same-runtime proof sweep
- `capacity`: turbo host only staircase that stops on first spill/offload or failure
- `spill`: turbo host only staircase that continues past the full-GPU limit and uses host RAM until failure
- `impact`: fast first-pass profile for old GPUs; one KV-memory cell plus one decode-throughput cell
- `quick`: `8k/16k`, concurrency `1/2`, `3` timed epochs
- `full`: `8k/16k/32k/64k`, concurrency `1/2/4`, `6` timed epochs
- `staircase`: convenience staircase preset
- `large-context`: descending long-context ladder with paired `residual_tail_tokens=0` and `128` rows unless explicitly overridden

## Legacy Host Compatibility

- baseline stock Ollama can be treated as `legacy`
- TurboQuant fork hosts should be treated as `request`
- legacy hosts never receive `kv_cache_type` or `kv_cache_backend`
- legacy hosts run default `f16` rows normally
- legacy hosts mark `q8_0`, `q4_0`, `tq25`, and `tq35` as unsupported instead of silently using host defaults
- if `--host-kv-support` is omitted, the first host defaults to `legacy` and later hosts default to `request`

## Claim Hierarchy

- Regression gate: `stock f16` vs `turbo f16`
- Causal TurboQuant claim: `turbo f16` vs `turbo tq25/tq35`
- Primary product-value claim: maximum full-GPU and maximum stable `context x concurrency`
- Adoption claim: `stock f16` vs `turbo tq35`

Do not use `baseline f16 vs turbo tq35` as the only claim. Use it as the end-user outcome summary alongside the same-runtime and capacity summaries.

## Recommended M40 First Pass

For old GPUs like Tesla M40, use `impact` first when you want a fast answer to:

- did KV / VRAM get smaller
- did throughput move

Example:

```bash
kvstress-bench \
  --hosts http://127.0.0.1:11438,http://127.0.0.1:11439 \
  --host-labels baseline,turbo \
  --model qwen3-30b:instruct2507-udq4kxl \
  --kv-modes f16,tq35 \
  --profile impact
```

## Host-Shell Flow

```bash
cd /path/to/ollama-main-turboquant

bash ./scripts/run-base-minimal-test-matrix.sh
bash ./scripts/run-turbo-minimal-test-matrix.sh
bash ./scripts/run-ab-minimal-test-matrix.sh
bash ./scripts/run-ab-ppl.sh
bash ./scripts/run-ab-agentic.sh
bash ./scripts/run-ab-full-large-context.sh
```

## How To Read Results

- `peak_vram_bytes`: the most direct full-GPU memory pressure signal
- `peak_host_ram_bytes`: how much host RAM was used once spill was allowed
- `prefill_tps`: prompt ingestion throughput
- `decode_tps`: generation throughput after prefill
- `live_kv_tokens_total`: approximate live KV occupancy across concurrent requests
- `full_gpu_residency`: only these rows belong in headline speed claims
- `spilled`: mixed CPU/GPU residency; valid for capacity and spill summaries, excluded from full-GPU speed claims
- `kv_path`: actual runtime path from the server
- `kv_algo_resolved`: must be `paper` for valid `tq25` / `tq35` benchmark rows
- `qjl_k_requested` / `qjl_k_effective`: whether the experimental K-side QJL lane was actually used
- `qjl_v_requested` / `qjl_v_effective`: V-side QJL requests should remain false/effective false in this checkout
- `v_reconstruction_compute_dtype`: supported V TurboQuant rows should report `fp32`
- `residual_tail_tokens`: compare `0` vs `128` on long-context profiles
- `niah_depth` / `niah_pass`: exact retrieval check under context pressure
- `corruption_class`: distinguishes empty output, punctuation repetition, malformed JSON, truncation, and other corruption-style failures
- `kl_divergence_vs_baseline`: sparse KLD if token-logprob overlap is sufficient
- `status=unsupported`: host cannot run that KV mode; skip it in performance claims
- `status=failed`: request or epoch failed; inspect `error`
- `success` is tri-state: `true`, `false`, or empty/null for unsupported
- memory/capacity summary: the main proof for “how much more live context can we hold?”

If `kv_path` is `dense-fallback`, that means TurboQuant may have been requested, but the server did not execute a TurboQuant fast path.

## Notes

- Primary claim language should be:
  - `stock f16 vs turbo f16` for fork regression
  - `turbo f16 vs turbo tq25/tq35` for same-runtime TurboQuant benefit
  - memory/capacity summary for maximum full-GPU and maximum stable `ctx x concurrency`
  - `stock f16 vs turbo tq35` only as the adoption/product outcome claim
- Use Flash Attention where required for quantized KV modes.
- Bigger differences should appear at `16k+` context and/or concurrency `>1`.
- GGUF weight quantization such as `Q4_K_M` is separate from KV cache mode.
- Experimental weight quantization fields are scaffold-only metadata in this checkout and do not indicate active model-weight compression.
