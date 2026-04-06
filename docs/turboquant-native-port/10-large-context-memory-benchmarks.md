# Large-Context Memory Benchmarks

Large-context native comparisons are driven by `kvstress-bench --profile large-context`.

The default descending ladder is:

- `1000000`
- `750000`
- `500000`
- `262144`
- `128000`
- `104000`
- `96000`
- `65536`
- `32768`
- `16384`

Every row keeps requested context separate from effective context and records fallback reason, RAM usage, VRAM usage, and residency kind.

Long-context profiles now also surface:

- `residual_tail_tokens` so tail/no-tail A/B rows can be compared directly
- `qjl_k_requested` / `qjl_k_effective`
- `qjl_v_requested` / `qjl_v_effective`
- `v_reconstruction_compute_dtype`
- `niah_depth` / `niah_pass`
- `corruption_class`
- `kl_divergence_vs_baseline` when sparse token-logprob overlap is sufficient

Residency kinds:

- `gpu-vram-only`
- `mixed-ram-vram`
- `cpu-assisted`
- `mmap-assisted`
- `unknown`

Primary large-context stress model from the installed set:

- `qwen35:udiq4xs`

Minimal host-shell large-context A/B run:

```bash
bash ./scripts/run-ab-full-large-context.sh
```

Current interpretation rules:

- stable wins require correctness first, then PPL / KLD / NIAH, then memory and throughput
- segmented-head and surface-aware rows are experimental
- experimental weight-quantization fields are scaffold-only metadata and do not imply active weight compression
