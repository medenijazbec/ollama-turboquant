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

$envConfigPath = Join-Path $resolvedRepoRoot "envconfig\config.go"
$llmServerPath = Join-Path $resolvedRepoRoot "llm\server.go"
$ggmlPath = Join-Path $resolvedRepoRoot "fs\ggml\ggml.go"
$backendPath = Join-Path $resolvedRepoRoot "ml\backend.go"
$runnerCachePath = Join-Path $resolvedRepoRoot "runner\ollamarunner\cache.go"
$llamaPath = Join-Path $resolvedRepoRoot "llama\llama.go"
$llmServerTestPath = Join-Path $resolvedRepoRoot "llm\server_test.go"
$ggmlTestPath = Join-Path $resolvedRepoRoot "fs\ggml\ggml_test.go"
$runnerCacheTestPath = Join-Path $resolvedRepoRoot "runner\ollamarunner\cache_test.go"
$llamaTestPath = Join-Path $resolvedRepoRoot "llama\llama_test.go"
$scorecardPath = Join-Path $resolvedRepoRoot "docs\turboquant_plan2_scorecard.md"
$testScriptPath = Join-Path $resolvedRepoRoot "scripts\test_turboquant_plan2.ps1"
$scoreScriptPath = Join-Path $resolvedRepoRoot "scripts\score_turboquant_plan2.ps1"

$envConfigText = Read-FileOrEmpty $envConfigPath
$llmServerText = Read-FileOrEmpty $llmServerPath
$ggmlText = Read-FileOrEmpty $ggmlPath
$backendText = Read-FileOrEmpty $backendPath
$runnerCacheText = Read-FileOrEmpty $runnerCachePath
$llamaText = Read-FileOrEmpty $llamaPath
$llmServerTestText = Read-FileOrEmpty $llmServerTestPath
$ggmlTestText = Read-FileOrEmpty $ggmlTestPath
$runnerCacheTestText = Read-FileOrEmpty $runnerCacheTestPath
$llamaTestText = Read-FileOrEmpty $llamaTestPath
$scorecardText = Read-FileOrEmpty $scorecardPath

$modeChecks = @(
    [pscustomobject]@{
        Name = "Env Description Updated"
        Weight = 15
        Passed = (Test-AllPatterns $envConfigText @("OLLAMA_KV_CACHE_TYPE", "tq25", "tq35", "tq3", "tq4"))
    },
    [pscustomobject]@{
        Name = "llm/server Normalization And Gating Tested"
        Weight = 20
        Passed = (
            (Test-AllPatterns $llmServerText @("newKVCacheMode", "selectLegacyKVCacheType", "selectEngineKVCacheType", "logAcceptedKVCacheType")) -and
            (Test-AllPatterns $llmServerTestText @("TestNewKVCacheMode", "TestSelectLegacyKVCacheType", "TestSelectEngineKVCacheType", "TestLogAcceptedKVCacheType"))
        )
    },
    [pscustomobject]@{
        Name = "GGML Support And Memory Tested"
        Weight = 20
        Passed = (
            (Test-AllPatterns $ggmlText @("SupportsKVCacheType", "KVCacheTypeIsQuantized", "kvCacheBytesPerElement", "tq25", "tq35")) -and
            (Test-AllPatterns $ggmlTestText @("TestSupportsKVCacheTypeTurboQuant", "TestKVCacheTypeIsQuantizedTurboQuant", "TestKVCacheBytesPerElementTurboQuant", "0.3125", "0.4375"))
        )
    },
    [pscustomobject]@{
        Name = "Runner DType Mapping Tested"
        Weight = 20
        Passed = (
            (Test-AllPatterns $runnerCacheText @("kvCacheTypeFromStr", "DTypeTQ25", "DTypeTQ35")) -and
            (Test-AllPatterns $runnerCacheTestText @('""', '"f16"', '"q8_0"', '"q4_0"', '"tq25"', '"tq35"', '"tq3"', '"tq4"'))
        )
    },
    [pscustomobject]@{
        Name = "Legacy Rejection Tested"
        Weight = 15
        Passed = (
            (Test-AllPatterns $llamaText @("legacy llama runner does not support TurboQuant KV cache type", "tq25", "tq35")) -and
            (Test-AllPatterns $llamaTestText @("TestNewContextParamsRejectsTurboQuant", "legacy llama runner", "tq35"))
        )
    },
    [pscustomobject]@{
        Name = "Logging Explicit And Testable"
        Weight = 10
        Passed = (
            (Test-AllPatterns $llmServerText @("using kv cache type", "kvCacheModeLogAttrs", "effective")) -and
            (Test-AllPatterns $llmServerTestText @("TestKVCacheModeLogAttrs", "TestLogAcceptedKVCacheType"))
        )
    }
)

$compatChecks = @(
    [pscustomobject]@{
        Name = "Existing Modes Covered"
        Weight = 25
        Passed = (Test-AllPatterns $runnerCacheTestText @('""', '"f16"', '"q8_0"', '"q4_0"'))
    },
    [pscustomobject]@{
        Name = "Alias Normalization Consistent"
        Weight = 25
        Passed = (
            (Test-AllPatterns $llmServerText @('case "tq3", "tq4":', 'return "tq35"')) -and
            (Test-AllPatterns $ggmlText @('case "tq3", "tq4":', 'return "tq35"')) -and
            (Test-AllPatterns $runnerCacheText @('case "tq3", "tq4":', 'return "tq35"')) -and
            (Test-AllPatterns $llamaText @('case "tq3", "tq4":', 'return "tq35"'))
        )
    },
    [pscustomobject]@{
        Name = "Legacy Rejection Explicit"
        Weight = 25
        Passed = (
            ($llamaText -match [regex]::Escape('legacy llama runner does not support TurboQuant KV cache type')) -and
            ($llamaText -notmatch [regex]::Escape('case "tq25"')) -and
            ($llamaText -notmatch [regex]::Escape('case "tq35"'))
        )
    },
    [pscustomobject]@{
        Name = "Docs And Scripts Defer Runtime To Plan 5"
        Weight = 25
        Passed = (
            (Test-Path $scorecardPath) -and
            (Test-Path $testScriptPath) -and
            (Test-Path $scoreScriptPath) -and
            (Test-AllPatterns $scorecardText @("deferred until Plan 5", "static", "pre", "post"))
        )
    }
)

foreach ($check in $modeChecks) {
    $check | Add-Member -NotePropertyName Score -NotePropertyValue (Get-Score -Weight $check.Weight -Passed $check.Passed)
}

foreach ($check in $compatChecks) {
    $check | Add-Member -NotePropertyName Score -NotePropertyValue (Get-Score -Weight $check.Weight -Passed $check.Passed)
}

$modeScore = ($modeChecks | Measure-Object -Property Score -Sum).Sum
$compatScore = ($compatChecks | Measure-Object -Property Score -Sum).Sum

$result = [pscustomobject]@{
    RepoRoot = $resolvedRepoRoot
    ModePlumbingCompletenessScore = $modeScore
    StaticCompatibilityScore = $compatScore
    ModeChecks = $modeChecks
    CompatibilityChecks = $compatChecks
}

$json = $result | ConvertTo-Json -Depth 6

Write-Output ("Mode Plumbing Completeness Score: {0}/100" -f $modeScore)
Write-Output ("Static Compatibility Score: {0}/100" -f $compatScore)
Write-Output ""
Write-Output "Mode Plumbing Breakdown:"
foreach ($check in $modeChecks) {
    Write-Output ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
}
Write-Output ""
Write-Output "Static Compatibility Breakdown:"
foreach ($check in $compatChecks) {
    Write-Output ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
}

if ($JsonPath) {
    Set-Content -Path $JsonPath -Value $json
}

if ($MarkdownPath) {
    $md = @(
        "# TurboQuant Plan 2 Scores",
        "",
        ("- Mode Plumbing Completeness Score: {0}/100" -f $modeScore),
        ("- Static Compatibility Score: {0}/100" -f $compatScore),
        "",
        "## Mode Plumbing Breakdown"
    )

    foreach ($check in $modeChecks) {
        $md += ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
    }

    $md += ""
    $md += "## Static Compatibility Breakdown"

    foreach ($check in $compatChecks) {
        $md += ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
    }

    Set-Content -Path $MarkdownPath -Value ($md -join "`r`n")
}

Write-Output ""
Write-Output "JSON:"
Write-Output $json
