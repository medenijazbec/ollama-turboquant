param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot "..")
)

$ErrorActionPreference = "Stop"

$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path

$checks = @(
    @{ Path = "kvcache\turboquant.go"; Patterns = @("getFastPathTensors", "SupportsTurboQuantFastPath()", "ctx.Input().FromBytes", "rowBytes") },
    @{ Path = "ml\backend\ggml\ggml.go"; Patterns = @("GGML_TYPE_OLLAMA_TQ25_KV", "GGML_TYPE_OLLAMA_TQ35_KV", "turboQuantAttentionScores", "using turboquant cpu attention fast path", "skipping turboquant cpu attention fast path") },
    @{ Path = "ml\backend\ggml\ggml\include\ggml.h"; Patterns = @("GGML_TYPE_OLLAMA_TQ25_KV", "GGML_TYPE_OLLAMA_TQ35_KV") },
    @{ Path = "ml\backend\ggml\ggml\src\ggml.c"; Patterns = @("ollama_tq25_kv", "ollama_tq35_kv") },
    @{ Path = "turboquant\decode.go"; Patterns = @("func ScoreEncodedVector", "residualDotCorrection") },
    @{ Path = "turboquant\decode_test.go"; Patterns = @("TestScoreEncodedVectorMatchesDecodedDot") },
    @{ Path = "kvcache\turboquant_test.go"; Patterns = @("TestTurboQuantCacheGetUsesCompressedKeyFastPath", "TestTurboQuantCacheFastPathFallsBackForInconsistentRows") },
    @{ Path = "ml\backend\ggml\ggml_test.go"; Patterns = @("TestTurboQuantDTypeRoundTrip", "TestSupportsTurboQuantFastPathRequiresFlashAttention", "TestTurboQuantScaledDotProductAttentionMatchesDenseReference") },
    @{ Path = "docs\turboquant_plan5_scorecard.md"; Patterns = @("05_backend_fastpath_and_validation.plan.txt", "ollama_turboquant_full_plan.txt", "ollama_turboquant_algorithms.txt", "Backend Fast Path Readiness Score", "Validation and Benchmark Coverage Score", "Plan 1", "Plan 4") },
    @{ Path = "scripts\benchmark_turboquant_plan5.ps1"; Patterns = @("tq25", "tq35", "f16", "q8_0", "q4_0") }
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
    Write-Error ("TurboQuant Plan 5 validation failed:`n- " + ($errors -join "`n- "))
    exit 1
}

$go = Get-Command go -ErrorAction SilentlyContinue
if ($null -ne $go) {
    Push-Location $resolvedRepoRoot
    try {
        & go test ./turboquant ./kvcache ./ml/backend/ggml
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Output "Go toolchain not available; skipped 'go test ./turboquant ./kvcache ./ml/backend/ggml' execution."
}

Write-Output "TurboQuant Plan 5 validation passed."
