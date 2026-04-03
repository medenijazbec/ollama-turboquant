# TurboQuant Docker Test Setup

This repository contains only the TurboQuant test deployment. The existing native Ollama service remains in the separate ADA deployment and continues to serve on:

- `http://127.0.0.1:11438`

The TurboQuant test deployment built from this repo serves on:

- `http://127.0.0.1:11439`

## Files

- `Dockerfile.sm52-turboquant`
  - custom CUDA 11.8 / sm_52 build for this TurboQuant fork
- `docker-compose.sm52-turboquant.yml`
  - standalone TurboQuant test deployment
- `Dockerfile.ollama-bench`
  - native Ollama benchmark container
- `docker-compose.ollama-bench.yml`
  - standalone benchmark runner container

## Start TurboQuant test container

```bash
cd /srv/share/ollama-benchmark/ollama-main-turboquant

docker compose -f docker-compose.sm52-turboquant.yml down
docker compose -f docker-compose.sm52-turboquant.yml build --pull
docker compose -f docker-compose.sm52-turboquant.yml up -d --force-recreate --remove-orphans
docker compose -f docker-compose.sm52-turboquant.yml logs -f --tail=200
```

## Stop TurboQuant test container

```bash
cd /srv/share/ollama-benchmark/ollama-main-turboquant
docker compose -f docker-compose.sm52-turboquant.yml down
```

## Logs

```bash
docker compose -f docker-compose.sm52-turboquant.yml logs -f --tail=200
```

## TurboQuant defaults

The test container is configured with:

- `OLLAMA_KV_CACHE_TYPE=tq35`
- `OLLAMA_FLASH_ATTENTION=1`
- `OLLAMA_DEBUG=DEBUG`
- `OLLAMA_LLM_LIBRARY=cuda_v11_avx`

Important: this deployment uses the same GPU-backed shape as your native coding Ollama service. The current TurboQuant implementation in this repo does not provide a GPU TurboQuant fast path yet. On this deployment, requesting `tq35` exercises the GPU production shape, but the runner falls back to dense KV when the backend cannot use the TurboQuant fast path. The benchmark output is labeled `turboquant-requested-gpu-dense-fallback` to make that explicit.

You can also run the new CLI flag directly inside the container:

```bash
docker compose -f docker-compose.sm52-turboquant.yml exec coding-ollama52-turboquant ollama run qwen35:udiq4xs --turboquant
docker compose -f docker-compose.sm52-turboquant.yml exec coding-ollama52-turboquant ollama run qwen35:udiq4xs --turboquant=tq25
docker compose -f docker-compose.sm52-turboquant.yml exec coding-ollama52-turboquant ollama run qwen35:udiq4xs --turboquant=off
```

## Native benchmark runner

Build and start the native benchmark container:

```bash
cd /srv/share/ollama-benchmark/ollama-main-turboquant

docker compose -f docker-compose.ollama-bench.yml down
docker compose -f docker-compose.ollama-bench.yml build --pull
docker compose -f docker-compose.ollama-bench.yml up -d --force-recreate --remove-orphans
```

These compose files now use different explicit project names, so:

- `docker-compose.sm52-turboquant.yml` controls only the TurboQuant test server
- `docker-compose.ollama-bench.yml` controls only the benchmark runner

`docker compose ... down` on one will no longer remove the other.

This container writes benchmark outputs to:

- `/srv/share/ollama-benchmark/results/test-native-ollama.txt`
- `/srv/share/ollama-benchmark/results/test-turboquant-ollama.txt`

Run the native test against the existing container on `11438`:

```bash
docker compose -f docker-compose.ollama-bench.yml exec ollama-bench sh /scripts/run-native-ollama-bench-qwen35.sh
```

Run the TurboQuant test against the test container on `11439`:

```bash
docker compose -f docker-compose.ollama-bench.yml exec ollama-bench sh /scripts/run-turboquant-ollama-bench-qwen35.sh
```

If you prefer the raw commands:

```bash
docker compose -f docker-compose.ollama-bench.yml exec ollama-bench sh -lc 'export OLLAMA_HOST=http://127.0.0.1:11438 && ollama-bench -model qwen35:udiq4xs -epochs 6 -warmup 1 -timeout 1800 -prompt-tokens 512 -max-tokens 64 -format benchstat -output /results/test-native-ollama.txt && cat /results/test-native-ollama.txt'
```

```bash
docker compose -f docker-compose.ollama-bench.yml exec ollama-bench sh -lc 'export OLLAMA_HOST=http://127.0.0.1:11439 && ollama-bench -model qwen35:udiq4xs -epochs 6 -warmup 1 -timeout 1800 -prompt-tokens 512 -max-tokens 64 -turboquant tq35 -format benchstat -output /results/test-turboquant-ollama.txt && cat /results/test-turboquant-ollama.txt'
```

The native benchmark path is the recommended one for Ollama because it uses the built-in Ollama API client and understands TurboQuant KV options directly.

## One-time cleanup for old fixed-name containers

If you created earlier revisions before these compose files stopped using `container_name`, remove the stale conflicting containers once:

```bash
docker rm -f coding-ollama52-turboquant ollama-bench 2>/dev/null || true
```

Then recreate both compose projects with the current files.
