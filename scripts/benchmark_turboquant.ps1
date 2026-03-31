$ErrorActionPreference = "Stop"

Write-Host "TurboQuant benchmark scaffold"
Write-Host "This script runs the codec microbenchmark and points to the full with/without TurboQuant benchmark workflow."
Write-Host "Examples:"
Write-Host "  ./ollama-bench -model gemma3 -epochs 6 -turboquant=off"
Write-Host "  ./ollama-bench -model gemma3 -epochs 6 -turboquant=tq35"
Write-Host "  ./ollama-bench -model gemma3 -epochs 6 -turboquant=tq25"

if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
  Write-Error "go is not available on PATH"
}

go test ./turboquant -bench BenchmarkEncodeDecodeTQ35 -benchmem
