param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot "..")
)

$ErrorActionPreference = "Stop"

$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path

$checks = @(
    @{ Path = "kvcache\causal.go"; Patterns = @("curLocs", "c.curLocs = make([]int, len(locs))") },
    @{ Path = "kvcache\turboquant.go"; Patterns = @("meta.curLocs[i]", "keyStride", "valueStride", "shiftPackedKeys", "permuteValueRows", "endIndex != math.MaxInt32", "ErrNotSupported") },
    @{ Path = "runner\ollamarunner\cache.go"; Patterns = @("WrapWithTurboQuant", "using turboquant kv cache", "kvCacheTypeFromStr") },
    @{ Path = "kvcache\turboquant_test.go"; Patterns = @("TestTurboQuantCacheStoreRoundTrip", "TestTurboQuantCacheMultiBatchAppend", "TestTurboQuantCacheCopyPrefixAndResume", "TestTurboQuantCacheRemoveTailClearsPackedEntries", "TestTurboQuantCacheRemoveMiddleShiftsKeys", "TestTurboQuantCacheSWAWindowBehavior", "TestTurboQuantCacheSWAMemResumeBehavior", "TestWrapWithTurboQuantPreservesNonCausalCaches", "keyDim != valueDim") },
    @{ Path = "runner\ollamarunner\cache_test.go"; Patterns = @("TestNewInputCacheWrapsTurboQuantCausalCaches", "TestNewInputCachePreservesWrapperNonCausalCaches", "TestNewInputCacheLeavesNonTurboQuantModesUnwrapped") },
    @{ Path = "docs\turboquant_plan4_scorecard.md"; Patterns = @("04_kvcache_phase_a_integration.plan.txt", "ollama_turboquant_full_plan.txt", "ollama_turboquant_algorithms.txt", "Plan 3", "deferred until Plan 5", "Phase A Cache Integration Completeness Score", "Cache Lifecycle Parity Score", "pre", "post") },
    @{ Path = "scripts\score_turboquant_plan4.ps1"; Patterns = @("Phase A Cache Integration Completeness Score", "Cache Lifecycle Parity Score", "Go toolchain") }
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
    Write-Error ("TurboQuant Plan 4 validation failed:`n- " + ($errors -join "`n- "))
    exit 1
}

$go = Get-Command go -ErrorAction SilentlyContinue
if ($null -ne $go) {
    Push-Location $resolvedRepoRoot
    try {
        & go test ./kvcache ./runner/ollamarunner
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Output "Go toolchain not available; skipped 'go test ./kvcache ./runner/ollamarunner' execution."
}

Write-Output "TurboQuant Plan 4 validation passed."
