param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot "..")
)

$ErrorActionPreference = "Stop"

Write-Host "TurboQuant Plan 5 benchmark entrypoint"
Write-Host "Modes: f16, q8_0, q4_0, tq35, tq25"
Write-Host "Metrics:"
Write-Host "- peak KV memory"
Write-Host "- total process memory"
Write-Host "- first-token latency"
Write-Host "- steady-state tok/s"
Write-Host "- mean absolute logit error"
Write-Host "- fixed-prompt output drift"
Write-Host ""
Write-Host "This scaffold is intended to be run after the Go toolchain and model assets are available locally."
Write-Host "It should be extended with concrete model invocations in environments that can run the full Plan 5 suite."
Write-Host ""
Write-Host "Suggested bench commands:"
Write-Host "  .\\ollama-bench -model gemma3 -epochs 6 -turboquant=off"
Write-Host "  .\\ollama-bench -model gemma3 -epochs 6 -turboquant=tq35"
Write-Host "  .\\ollama-bench -model gemma3 -epochs 6 -turboquant=tq25"

$legacyScript = Join-Path $PSScriptRoot "benchmark_turboquant.ps1"
if (Test-Path $legacyScript) {
    Write-Host "Codec microbenchmark scaffold:"
    Write-Host "powershell -File $legacyScript"
}
