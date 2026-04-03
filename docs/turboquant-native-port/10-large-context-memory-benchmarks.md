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

Residency kinds:

- `gpu-vram-only`
- `mixed-ram-vram`
- `cpu-assisted`
- `mmap-assisted`
- `unknown`

Primary large-context stress model from the installed set:

- `qwen35:udiq4xs`

Minimal large-context A/B run:

```bash
docker compose -f docker-compose.ollama-bench.yml exec ollama-bench \
  /scripts/run-ab-full-large-context.sh
```
