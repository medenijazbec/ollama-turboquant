# Native TurboQuant Test Matrix

This checkpoint turns `kvstress-bench` into the primary native TurboQuant test matrix runner.

Default A/B layout:

- baseline host: `http://127.0.0.1:11438`
- TurboQuant host: `http://127.0.0.1:11439`

Default minimal models from the installed server set:

| Model | Role |
|---|---|
| `qwen3.5-9b:udq4kxl` | dense 9B minimal A/B matrix |
| `deepseek-r1-distill-qwen7:q4km` | correctness, recall, corruption, prompt-file regression |
| `qwen35:udiq4xs` | large-context memory and fit-ceiling target |
| `qwen3.5-27b-heretic:q80` | larger dense FA and throughput target |
| `Qwen3-30B-A3B-abliterated-erotic-i1:Q6_K` | hybrid/MoE hook |

Implemented harness categories:

- context scaling / fit ceiling
- recall-under-distance
- corruption and prompt-file regression
- asymmetric K/V verification
- Flash Attention gating reporting
- staged RAM/VRAM telemetry with residency classification
- Markdown, CSV, and JSONL outputs

Scaffold-only in this checkpoint:

- multimodal / mmproj placement hooks
- very large 397B sweeps unless explicitly requested

Minimal A/B command:

```bash
docker compose -f docker-compose.ollama-bench.yml exec ollama-bench \
  /scripts/run-ab-minimal-test-matrix.sh
```
