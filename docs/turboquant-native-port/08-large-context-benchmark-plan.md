# Checkpoint 08

## Goal

Checkpoint 08 adds a large-context benchmark and validation lane around `cmd/benchkv` with:

- a dynamic context ladder
- mixed RAM + VRAM telemetry
- requested-vs-effective K/V and context truth
- long-context correctness guards
- checkpoint-visible V-path precision audit notes

## Target benchmark behavior

The large-context lane now attempts contexts in this order until one fits:

1. `1000000`
2. `750000`
3. `500000`
4. `262144`
5. `128000`
6. `64000`
7. `32768`

A rung is only treated as a fit when:

- model load succeeds
- prompt prefill succeeds
- final metrics are returned
- at least the configured minimum decode sample succeeds

The benchmark records:

- `requested_num_ctx`
- `attempted_num_ctx`
- `effective_num_ctx`
- `requested_context_top_rung`
- `context_ladder_index`
- `context_fallback_reason`
- `context_fallback_detail`
- `context_fallback_stage`
- `context_fallback_class`
- `ladder_rejected_rungs`

## Large-context workloads

Checkpoint 08 adds these workload lanes:

- `fit-ceiling`
- `long-context-recall`
- `long-json-retention`
- `prompt-file-regression`
- `decode-corruption-guard`

It also reuses `prefill-heavy` for large prefill throughput rows.

## Mode matrix

The large-context profile uses these requested K/V combinations by default:

- `f16/f16`
- `q8_0/q8_0`
- `q8_0/tq35`
- `tq35/tq35`

`--turbo-mode=tq25` switches the mixed and symmetric TurboQuant rows into the explicit experimental lane.

## Telemetry additions

Checkpoint 08 extends benchmark rows with:

- per-GPU used/free VRAM maps
- total visible VRAM
- visible GPU count
- process VRAM when available
- host RAM used
- peak host RAM
- peak host RAM delta
- heuristic host-assist flags:
  - `used_host_assist`
  - `used_mmap`
  - `used_cpu_assist`

Telemetry sources still follow:

- GPU: `nvml` -> `nvidia-smi` -> `unavailable`
- host RAM: `proc-meminfo` -> `free` -> `unavailable`

## Requested vs effective truth

Benchmark output keeps requested and effective state separate for both KV mode and context:

- requested K/V comes from the benchmark mode matrix
- effective K/V comes from runtime metrics
- requested context comes from the ladder target
- effective context comes from the successful runtime row

Rollout or performance claims should use effective values, not requested values.

## V-path precision audit

Checkpoint 08 explicitly audited the CPU/reference V reconstruction sites in:

- `turboquant/decode.go`
- `turboquant/rotation.go`
- `turboquant/residual_qjl.go`
- `kvcache/turboquant.go`

Current result:

- the reference decode/reconstruct path already accumulates in `float32`
- inverse rotation accumulation in `ApplyInverseRotation` is `float32`
- residual reconstruction accumulation in `reconstructResidual` is `float32`
- no backend-native half-accumulate V reconstruction path was promoted in this checkpoint

Remaining risk:

- any future backend-specific mirror of these paths must preserve FP32 accumulation explicitly
- long-generation corruption checks remain necessary because decode corruption can still hide behind superficially successful throughput runs

## Long-context correctness guards

Checkpoint 08 fully wires:

- `long-context-recall`
- `prompt-file-regression`
- `decode-corruption-guard`

`long-json-retention` is present in the schema and workload registry, but remains scaffolded unless a concrete harness is added later.

Corruption markers currently look for:

- repeated slash / question-mark patterns
- malformed JSON brace balance
- degenerate repeated-token runs

## Non-power-of-2 head-dim scaffold

Checkpoint 08 does not enable non-power-of-2 native rollout.

It adds an experimental scaffold for segmented head dims such as:

- `576 = 256 + 256 + 64`

Current status:

- metadata scaffold exists only
- the lane remains experimental
- no global native enablement was added

## Intended benchmark targets

Primary intended targets for this lane:

- `deepseek-r1:7b` / equivalent `q4_K_M` 128K-class target
- `Qwen3.5-35B-A3B` Unsloth dynamic GGUF around the 17-18 GiB class

The harness does not hardcode those tags. It records what the local runtime actually fit.

## Known limitations

- 1M context is treated as a stretch target and is only valid if the ladder proves it
- YaRN/stretch support is reporting-only in this checkpoint
- host-assist classification remains heuristic
- `long-json-retention` is still scaffolded if no concrete fixture is supplied
