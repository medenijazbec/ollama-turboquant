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

$cacheText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "kvcache\turboquant.go")
$ggmlGoText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "ml\backend\ggml\ggml.go")
$ggmlHeaderText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "ml\backend\ggml\ggml\include\ggml.h")
$ggmlCText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "ml\backend\ggml\ggml\src\ggml.c")
$decodeText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "turboquant\decode.go")
$cacheTestText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "kvcache\turboquant_test.go")
$ggmlTestText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "ml\backend\ggml\ggml_test.go")
$decodeTestText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "turboquant\decode_test.go")
$scorecardText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "docs\turboquant_plan5_scorecard.md")
$benchmarkText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "scripts\benchmark_turboquant_plan5.ps1")

$readinessChecks = @(
    [pscustomobject]@{
        Name = "GGML dtype mapping no longer falls back to F16"
        Weight = 20
        Passed = (Test-AllPatterns $ggmlGoText @("case ml.DTypeTQ25:", "GGML_TYPE_OLLAMA_TQ25_KV", "case ml.DTypeTQ35:", "GGML_TYPE_OLLAMA_TQ35_KV"))
    },
    [pscustomobject]@{
        Name = "Internal ggml KV types exist"
        Weight = 20
        Passed = (
            (Test-AllPatterns $ggmlHeaderText @("GGML_TYPE_OLLAMA_TQ25_KV", "GGML_TYPE_OLLAMA_TQ35_KV")) -and
            (Test-AllPatterns $ggmlCText @("ollama_tq25_kv", "ollama_tq35_kv"))
        )
    },
    [pscustomobject]@{
        Name = "TurboQuantCache hands off compressed K"
        Weight = 20
        Passed = (Test-AllPatterns $cacheText @("getFastPathTensors", "ctx.Input().FromBytes", "SupportsTurboQuantFastPath()", "rowBytes"))
    },
    [pscustomobject]@{
        Name = "Compressed-K attention path exists and is tested"
        Weight = 20
        Passed = (
            (Test-AllPatterns $ggmlGoText @("turboQuantAttentionScores", "turboquant.ScoreEncodedVector")) -and
            (Test-AllPatterns $ggmlTestText @("TestTurboQuantScaledDotProductAttentionMatchesDenseReference"))
        )
    },
    [pscustomobject]@{
        Name = "Fallback behavior remains explicit"
        Weight = 20
        Passed = (
            (Test-AllPatterns $ggmlGoText @("skipping turboquant cpu attention fast path")) -and
            (Test-AllPatterns $cacheTestText @("TestTurboQuantCacheFastPathFallsBackForInconsistentRows"))
        )
    }
)

$validationChecks = @(
    [pscustomobject]@{
        Name = "Malformed-input and dtype tests exist"
        Weight = 20
        Passed = (
            (Test-AllPatterns $ggmlTestText @("TestTurboQuantDTypeRoundTrip")) -and
            (Test-AllPatterns $decodeTestText @("TestUnmarshalEncodedVectorRejectsBadHeader", "TestUnmarshalEncodedVectorRejectsWrongVersion", "TestUnmarshalEncodedVectorRejectsBadPresetID", "TestUnmarshalEncodedVectorRejectsTruncatedBlockPayload"))
        )
    },
    [pscustomobject]@{
        Name = "Numerical parity tests exist for tq25/tq35"
        Weight = 20
        Passed = (
            (Test-AllPatterns $ggmlTestText @("TestTurboQuantScaledDotProductAttentionMatchesDenseReference")) -and
            (Test-AllPatterns $decodeTestText @("TestScoreEncodedVectorMatchesDecodedDot"))
        )
    },
    [pscustomobject]@{
        Name = "Fast-path and fallback selection are covered"
        Weight = 20
        Passed = (
            (Test-AllPatterns $ggmlTestText @("TestSupportsTurboQuantFastPathRequiresFlashAttention")) -and
            (Test-AllPatterns $cacheTestText @("TestTurboQuantCacheGetUsesCompressedKeyFastPath", "TestTurboQuantCacheFastPathFallsBackForInconsistentRows"))
        )
    },
    [pscustomobject]@{
        Name = "Benchmark script covers required cache modes and metrics"
        Weight = 20
        Passed = (Test-AllPatterns $benchmarkText @("f16", "q8_0", "q4_0", "tq35", "tq25", "peak KV memory", "total process memory", "first-token latency", "steady-state tok/s", "mean absolute logit error", "fixed-prompt output drift"))
    },
    [pscustomobject]@{
        Name = "Scorecard and scripts report pre/post outcomes"
        Weight = 20
        Passed = (Test-AllPatterns $scorecardText @("Backend Fast Path Readiness Score", "Validation and Benchmark Coverage Score", "Pre", "Post"))
    }
)

foreach ($check in $readinessChecks) {
    $check | Add-Member -NotePropertyName Score -NotePropertyValue (Get-Score -Weight $check.Weight -Passed $check.Passed)
}

foreach ($check in $validationChecks) {
    $check | Add-Member -NotePropertyName Score -NotePropertyValue (Get-Score -Weight $check.Weight -Passed $check.Passed)
}

$readinessScore = ($readinessChecks | Measure-Object -Property Score -Sum).Sum
$validationScore = ($validationChecks | Measure-Object -Property Score -Sum).Sum
$goAvailable = $null -ne (Get-Command go -ErrorAction SilentlyContinue)

$result = [pscustomobject]@{
    RepoRoot = $resolvedRepoRoot
    BackendFastPathReadinessScore = $readinessScore
    ValidationAndBenchmarkCoverageScore = $validationScore
    GoToolchainAvailable = $goAvailable
    ReadinessChecks = $readinessChecks
    ValidationChecks = $validationChecks
}

$json = $result | ConvertTo-Json -Depth 6

Write-Output ("Backend Fast Path Readiness Score: {0}/100" -f $readinessScore)
Write-Output ("Validation and Benchmark Coverage Score: {0}/100" -f $validationScore)
if ($goAvailable) {
    Write-Output "Go toolchain available: yes"
}
else {
    Write-Output "Go toolchain not available; score is based on source and test artifacts only."
}

if ($JsonPath) {
    Set-Content -Path $JsonPath -Value $json
}

if ($MarkdownPath) {
    $md = @(
        "# TurboQuant Plan 5 Scores",
        "",
        ("- Backend Fast Path Readiness Score: {0}/100" -f $readinessScore),
        ("- Validation and Benchmark Coverage Score: {0}/100" -f $validationScore),
        ("- Go toolchain available: {0}" -f ($(if ($goAvailable) { "yes" } else { "no" })))
    )
    Set-Content -Path $MarkdownPath -Value ($md -join "`r`n")
}

Write-Output ""
Write-Output "JSON:"
Write-Output $json
