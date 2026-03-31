param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot "..")
)

$ErrorActionPreference = "Stop"

$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path

$checks = @(
    @{ Path = "envconfig\config.go"; Patterns = @("OLLAMA_KV_CACHE_TYPE", "tq25", "tq35", "tq3", "tq4") },
    @{ Path = "llm\server.go"; Patterns = @("newKVCacheMode", "selectLegacyKVCacheType", "selectEngineKVCacheType", "logAcceptedKVCacheType", "using kv cache type") },
    @{ Path = "fs\ggml\ggml.go"; Patterns = @("SupportsKVCacheType", "KVCacheTypeIsQuantized", "kvCacheBytesPerElement", "0.3125", "0.4375") },
    @{ Path = "runner\ollamarunner\cache.go"; Patterns = @("kvCacheTypeFromStr", "DTypeTQ25", "DTypeTQ35") },
    @{ Path = "llama\llama.go"; Patterns = @("legacy llama runner does not support TurboQuant KV cache type", "tq25", "tq35") },
    @{ Path = "llm\server_test.go"; Patterns = @("TestNewKVCacheMode", "TestSelectLegacyKVCacheType", "TestSelectEngineKVCacheType", "TestLogAcceptedKVCacheType") },
    @{ Path = "fs\ggml\ggml_test.go"; Patterns = @("TestSupportsKVCacheTypeTurboQuant", "TestKVCacheTypeIsQuantizedTurboQuant", "TestKVCacheBytesPerElementTurboQuant") },
    @{ Path = "runner\ollamarunner\cache_test.go"; Patterns = @('"f16"', '"q8_0"', '"q4_0"', '"tq25"', '"tq35"', '"tq3"', '"tq4"') },
    @{ Path = "llama\llama_test.go"; Patterns = @("TestNewContextParamsRejectsTurboQuant", "legacy llama runner", "tq35") },
    @{ Path = "docs\turboquant_plan2_scorecard.md"; Patterns = @("deferred until Plan 5", "Mode Plumbing Completeness Score", "Static Compatibility Score", "pre", "post") },
    @{ Path = "scripts\score_turboquant_plan2.ps1"; Patterns = @("Mode Plumbing Completeness Score", "Static Compatibility Score") }
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
    Write-Error ("TurboQuant Plan 2 validation failed:`n- " + ($errors -join "`n- "))
    exit 1
}

Write-Output "TurboQuant Plan 2 validation passed."
