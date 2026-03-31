param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot ".."),
    [string]$JsonPath,
    [string]$MarkdownPath
)

$ErrorActionPreference = "Stop"

function Read-FileOrEmpty {
    param([string]$Path)

    if (Test-Path $Path) {
        return Get-Content -Raw $Path
    }

    return ""
}

function Test-AllPatterns {
    param(
        [string]$Text,
        [string[]]$Patterns
    )

    foreach ($pattern in $Patterns) {
        if ($Text -notmatch [regex]::Escape($pattern)) {
            return $false
        }
    }

    return $true
}

function Get-Score {
    param(
        [int]$Weight,
        [bool]$Passed
    )

    if ($Passed) {
        return $Weight
    }

    return 0
}

$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path

$causalPath = Join-Path $resolvedRepoRoot "kvcache\causal.go"
$turboquantCachePath = Join-Path $resolvedRepoRoot "kvcache\turboquant.go"
$runnerCachePath = Join-Path $resolvedRepoRoot "runner\ollamarunner\cache.go"
$turboquantCacheTestPath = Join-Path $resolvedRepoRoot "kvcache\turboquant_test.go"
$runnerCacheTestPath = Join-Path $resolvedRepoRoot "runner\ollamarunner\cache_test.go"
$scorecardPath = Join-Path $resolvedRepoRoot "docs\turboquant_plan4_scorecard.md"
$testScriptPath = Join-Path $resolvedRepoRoot "scripts\test_turboquant_plan4.ps1"
$scoreScriptPath = Join-Path $resolvedRepoRoot "scripts\score_turboquant_plan4.ps1"

$causalText = Read-FileOrEmpty $causalPath
$turboquantCacheText = Read-FileOrEmpty $turboquantCachePath
$runnerCacheText = Read-FileOrEmpty $runnerCachePath
$turboquantCacheTestText = Read-FileOrEmpty $turboquantCacheTestPath
$runnerCacheTestText = Read-FileOrEmpty $runnerCacheTestPath
$scorecardText = Read-FileOrEmpty $scorecardPath

$integrationChecks = @(
    [pscustomobject]@{
        Name = "Runner Wiring And Preset Wrapping"
        Weight = 20
        Passed = (
            (Test-AllPatterns $runnerCacheText @("WrapWithTurboQuant", "kvcachePreset", "using turboquant kv cache")) -and
            (Test-AllPatterns $runnerCacheTestText @("TestNewInputCacheWrapsTurboQuantCausalCaches", "TestNewInputCachePreservesWrapperNonCausalCaches", "TestNewInputCacheLeavesNonTurboQuantModesUnwrapped"))
        )
    },
    [pscustomobject]@{
        Name = "Put Encodes Both K And V"
        Weight = 20
        Passed = (
            (Test-AllPatterns $turboquantCacheText @("keyStride", "valueStride", "encodeVectorBytes", "meta.curLocs[i]")) -and
            (Test-AllPatterns $turboquantCacheTestText @("TestTurboQuantCacheStoreRoundTrip", "value shape", "keyDim != valueDim"))
        )
    },
    [pscustomobject]@{
        Name = "Get Decodes Active Range And Preserves Layout"
        Weight = 20
        Passed = (
            (Test-AllPatterns $turboquantCacheText @("permuteValueRows", "ctx.Input().FromFloats", "c.meta.curCellRange.min", "c.meta.curMask")) -and
            (Test-AllPatterns $turboquantCacheTestText @("TestTurboQuantCacheStoreRoundTrip", "TestTurboQuantCacheMultiBatchAppend", "PermutedV"))
        )
    },
    [pscustomobject]@{
        Name = "CopyPrefix CanResume And Remove Reach Parity"
        Weight = 20
        Passed = (
            (Test-AllPatterns $turboquantCacheText @("CopyPrefix", "CanResume", "shiftPackedKeys", "sweepReleasedCells")) -and
            (Test-AllPatterns $turboquantCacheTestText @("TestTurboQuantCacheCopyPrefixAndResume", "TestTurboQuantCacheRemoveTailClearsPackedEntries", "TestTurboQuantCacheRemoveMiddleShiftsKeys", "TestTurboQuantCacheRemoveMiddleRequiresShiftSupport"))
        )
    },
    [pscustomobject]@{
        Name = "Plan 4 Artifacts Defer Runtime To Plan 5"
        Weight = 20
        Passed = (
            (Test-Path $scorecardPath) -and
            (Test-Path $testScriptPath) -and
            (Test-Path $scoreScriptPath) -and
            (Test-AllPatterns $scorecardText @("deferred until Plan 5", "static + unit + cache-integration", "Phase A Cache Integration Completeness Score", "Cache Lifecycle Parity Score"))
        )
    }
)

$parityChecks = @(
    [pscustomobject]@{
        Name = "Store Get Covers Presets And PermutedV"
        Weight = 20
        Passed = (Test-AllPatterns $turboquantCacheTestText @("TestTurboQuantCacheStoreRoundTrip", "PresetTQ25", "PresetTQ35", "runPermutedVariants"))
    },
    [pscustomobject]@{
        Name = "Value Stride Regression Covered"
        Weight = 20
        Passed = (Test-AllPatterns $turboquantCacheTestText @("keyDim != valueDim", "value shape = %v, want [3 2 2]"))
    },
    [pscustomobject]@{
        Name = "Middle Delete Shift And Tail Cleanup Covered"
        Weight = 25
        Passed = (Test-AllPatterns $turboquantCacheTestText @("TestTurboQuantCacheRemoveTailClearsPackedEntries", "TestTurboQuantCacheRemoveMiddleShiftsKeys", "TestTurboQuantCacheRemoveMiddleRequiresShiftSupport"))
    },
    [pscustomobject]@{
        Name = "Wrapper And Encoder Preservation Covered"
        Weight = 15
        Passed = (
            (Test-AllPatterns $turboquantCacheTestText @("TestWrapWithTurboQuantPreservesNonCausalCaches")) -and
            (Test-AllPatterns $runnerCacheTestText @("TestNewInputCachePreservesWrapperNonCausalCaches"))
        )
    },
    [pscustomobject]@{
        Name = "SWA And Resume Behavior Covered"
        Weight = 20
        Passed = (Test-AllPatterns $turboquantCacheTestText @("TestTurboQuantCacheSWAWindowBehavior", "TestTurboQuantCacheSWAMemResumeBehavior"))
    }
)

foreach ($check in $integrationChecks) {
    $check | Add-Member -NotePropertyName Score -NotePropertyValue (Get-Score -Weight $check.Weight -Passed $check.Passed)
}

foreach ($check in $parityChecks) {
    $check | Add-Member -NotePropertyName Score -NotePropertyValue (Get-Score -Weight $check.Weight -Passed $check.Passed)
}

$integrationScore = ($integrationChecks | Measure-Object -Property Score -Sum).Sum
$parityScore = ($parityChecks | Measure-Object -Property Score -Sum).Sum
$goAvailable = $null -ne (Get-Command go -ErrorAction SilentlyContinue)

$result = [pscustomobject]@{
    RepoRoot = $resolvedRepoRoot
    PhaseACacheIntegrationCompletenessScore = $integrationScore
    CacheLifecycleParityScore = $parityScore
    GoToolchainAvailable = $goAvailable
    IntegrationChecks = $integrationChecks
    ParityChecks = $parityChecks
}

$json = $result | ConvertTo-Json -Depth 6

Write-Output ("Phase A Cache Integration Completeness Score: {0}/100" -f $integrationScore)
Write-Output ("Cache Lifecycle Parity Score: {0}/100" -f $parityScore)
if ($goAvailable) {
    Write-Output "Go toolchain available: yes"
}
else {
    Write-Output "Go toolchain not available; score is based on source and test artifacts only."
}
Write-Output ""
Write-Output "Phase A Integration Breakdown:"
foreach ($check in $integrationChecks) {
    Write-Output ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
}
Write-Output ""
Write-Output "Cache Lifecycle Breakdown:"
foreach ($check in $parityChecks) {
    Write-Output ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
}

if ($JsonPath) {
    Set-Content -Path $JsonPath -Value $json
}

if ($MarkdownPath) {
    $md = @(
        "# TurboQuant Plan 4 Scores",
        "",
        ("- Phase A Cache Integration Completeness Score: {0}/100" -f $integrationScore),
        ("- Cache Lifecycle Parity Score: {0}/100" -f $parityScore),
        ("- Go toolchain available: {0}" -f ($(if ($goAvailable) { "yes" } else { "no" }))),
        "",
        "## Phase A Integration Breakdown"
    )

    foreach ($check in $integrationChecks) {
        $md += ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
    }

    $md += ""
    $md += "## Cache Lifecycle Breakdown"

    foreach ($check in $parityChecks) {
        $md += ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
    }

    Set-Content -Path $MarkdownPath -Value ($md -join "`r`n")
}

Write-Output ""
Write-Output "JSON:"
Write-Output $json
