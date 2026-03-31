param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot "..")
)

$ErrorActionPreference = "Stop"

$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path

$checks = @(
    @{ Path = "api\types.go"; Patterns = @('KVCacheType string `json:"kv_cache_type,omitempty"`') },
    @{ Path = "api\types_test.go"; Patterns = @("TestKVCacheTypeParsingFromJSON", "TestKVCacheTypeFormatParams") },
    @{ Path = "server\routes_options_test.go"; Patterns = @("TestModelOptionsKVCacheTypePriority") },
    @{ Path = "llm\server.go"; Patterns = @("resolveKVCacheMode", "opts.KVCacheType", "envconfig.KvCacheType()") },
    @{ Path = "llm\server_test.go"; Patterns = @("TestResolveKVCacheMode") },
    @{ Path = "cmd\cmd.go"; Patterns = @("normalizeTurboQuantFlag", 'Lookup("turboquant").NoOptDefVal = "tq35"', 'opts.Options["kv_cache_type"]', 'warning: --turboquant is ignored for embedding models', 'warning: --turboquant is ignored for image generation models') },
    @{ Path = "cmd\cmd_test.go"; Patterns = @("TestNormalizeTurboQuantFlag", "TestLoadOrUnloadModelIncludesKVCacheTypeOption", "TestRunHandlerTurboQuantAddsGenerateOption", "TestRunHandlerTurboQuantIgnoredForEmbeddingModels") },
    @{ Path = "cmd\bench\bench.go"; Patterns = @("normalizeTurboQuantFlag", 'flag.String("turboquant"', 'options["kv_cache_type"]', "| KV:", "| Path:") },
    @{ Path = "cmd\bench\bench_test.go"; Patterns = @("TestNormalizeTurboQuantFlag", "TestBuildGenerateRequestIncludesKVCacheType") },
    @{ Path = "docs\openapi.yaml"; Patterns = @("kv_cache_type:") },
    @{ Path = "docs\api.md"; Patterns = @("kv_cache_type", "tq35") },
    @{ Path = "docs\cli.mdx"; Patterns = @("--turboquant", "--turboquant=tq25", "--turboquant=off") },
    @{ Path = "docs\modelfile.mdx"; Patterns = @("PARAMETER kv_cache_type tq35", "PARAMETER kv_cache_type tq25") },
    @{ Path = "docs\turboquant.mdx"; Patterns = @("TurboQuant is KV-cache compression only", "ollama run gemma3 --turboquant", "options.kv_cache_type", "GGUF", "Results Template") },
    @{ Path = "docs\docs.json"; Patterns = @('"/turboquant"') },
    @{ Path = "docs\turboquant_completion_matrix.md"; Patterns = @("TurboQuant Completion Matrix", "Productization", "Complete") },
    @{ Path = "docs\turboquant_completion_scorecard.md"; Patterns = @("Implementation Completeness Score", "Product Surface Completeness Score") },
    @{ Path = "scripts\score_turboquant_completion.ps1"; Patterns = @("Implementation Completeness Score", "Product Surface Completeness Score") }
)

$errors = New-Object System.Collections.Generic.List[string]

foreach ($check in $checks) {
    $fullPath = Join-Path $resolvedRepoRoot $check.Path
    if (-not (Test-Path $fullPath)) {
        $errors.Add("Missing required file: $($check.Path)")
        continue
    }

    $text = Get-Content -Raw $fullPath
    foreach ($pattern in $check.Patterns) {
        if ($text -notmatch [regex]::Escape($pattern)) {
            $errors.Add("Missing required pattern '$pattern' in $($check.Path)")
        }
    }
}

if ($errors.Count -gt 0) {
    Write-Error ("TurboQuant completion validation failed:`n- " + ($errors -join "`n- "))
    exit 1
}

$go = Get-Command go -ErrorAction SilentlyContinue
if ($null -ne $go) {
    Push-Location $resolvedRepoRoot
    try {
        & go test ./api ./cmd ./server ./llm ./kvcache ./ml/backend/ggml ./turboquant
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Output "Go toolchain not available; skipped 'go test ./api ./cmd ./server ./llm ./kvcache ./ml/backend/ggml ./turboquant' execution."
}

Write-Output "TurboQuant completion validation passed."
