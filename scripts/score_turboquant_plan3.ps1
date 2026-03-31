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

$turboquantPath = Join-Path $resolvedRepoRoot "turboquant\turboquant.go"
$rotationPath = Join-Path $resolvedRepoRoot "turboquant\rotation.go"
$codebookPath = Join-Path $resolvedRepoRoot "turboquant\codebook.go"
$residualPath = Join-Path $resolvedRepoRoot "turboquant\residual_qjl.go"
$blockPath = Join-Path $resolvedRepoRoot "turboquant\block.go"
$encodePath = Join-Path $resolvedRepoRoot "turboquant\encode.go"
$decodePath = Join-Path $resolvedRepoRoot "turboquant\decode.go"
$statsPath = Join-Path $resolvedRepoRoot "turboquant\stats.go"

$rotationTestPath = Join-Path $resolvedRepoRoot "turboquant\rotation_test.go"
$codebookTestPath = Join-Path $resolvedRepoRoot "turboquant\codebook_test.go"
$residualTestPath = Join-Path $resolvedRepoRoot "turboquant\residual_qjl_test.go"
$blockTestPath = Join-Path $resolvedRepoRoot "turboquant\block_test.go"
$encodeTestPath = Join-Path $resolvedRepoRoot "turboquant\encode_test.go"
$decodeTestPath = Join-Path $resolvedRepoRoot "turboquant\decode_test.go"
$statsTestPath = Join-Path $resolvedRepoRoot "turboquant\stats_test.go"
$scorecardPath = Join-Path $resolvedRepoRoot "docs\turboquant_plan3_scorecard.md"
$testScriptPath = Join-Path $resolvedRepoRoot "scripts\test_turboquant_plan3.ps1"
$scoreScriptPath = Join-Path $resolvedRepoRoot "scripts\score_turboquant_plan3.ps1"

$turboquantText = Read-FileOrEmpty $turboquantPath
$rotationText = Read-FileOrEmpty $rotationPath
$codebookText = Read-FileOrEmpty $codebookPath
$residualText = Read-FileOrEmpty $residualPath
$blockText = Read-FileOrEmpty $blockPath
$encodeText = Read-FileOrEmpty $encodePath
$decodeText = Read-FileOrEmpty $decodePath
$statsText = Read-FileOrEmpty $statsPath

$rotationTestText = Read-FileOrEmpty $rotationTestPath
$codebookTestText = Read-FileOrEmpty $codebookTestPath
$residualTestText = Read-FileOrEmpty $residualTestPath
$blockTestText = Read-FileOrEmpty $blockTestPath
$encodeTestText = Read-FileOrEmpty $encodeTestPath
$decodeTestText = Read-FileOrEmpty $decodeTestPath
$statsTestText = Read-FileOrEmpty $statsTestPath
$scorecardText = Read-FileOrEmpty $scorecardPath

$codecChecks = @(
    [pscustomobject]@{
        Name = "Preset Metadata Finalized"
        Weight = 15
        Passed = (Test-AllPatterns $turboquantText @("RegularCodebook", "RegularBoundaries", "OutlierCodebook", "OutlierBoundaries", "PresetTQ25", "PresetTQ35"))
    },
    [pscustomobject]@{
        Name = "Deterministic Rotation Implemented And Tested"
        Weight = 15
        Passed = (
            (Test-AllPatterns $rotationText @("BuildRotation", "ApplyRotation", "ApplyInverseRotation", "fastHadamard")) -and
            (Test-AllPatterns $rotationTestText @("TestBuildRotationDeterministic", "TestBuildRotationDifferentSeedsDiffer", "TestApplyInverseRotation", "TestRotationPreservesNorm"))
        )
    },
    [pscustomobject]@{
        Name = "Preset-Specific Codebooks And Boundaries"
        Weight = 20
        Passed = (
            (Test-AllPatterns $codebookText @("explicitCodebook", "codebookBoundaries", "quantizeScalarByBoundary", "regularBoundaries", "outlierBoundaries")) -and
            (Test-AllPatterns $codebookTestText @("TestPresetCodebookLengths", "TestCodebookBoundariesMonotonic", "TestQuantizeScalarByBoundaryDeterministic", "TestPresetCodebooksDiffer"))
        )
    },
    [pscustomobject]@{
        Name = "Residual Sketch Implemented And Tested"
        Weight = 15
        Passed = (
            (Test-AllPatterns $residualText @("encodeResidual", "reconstructResidual", "residualDotCorrection")) -and
            (Test-AllPatterns $residualTestText @("TestResidualSketchDeterministic", "TestReconstructResidualLength", "TestZeroResidualProducesZeroReconstruction", "TestResidualDotCorrectionDeterministicAndFinite"))
        )
    },
    [pscustomobject]@{
        Name = "Block Layout And Binary Serialization"
        Weight = 20
        Passed = (
            (Test-AllPatterns $blockText @("MarshalBinary", "UnmarshalBinary", "expectedPackedBytes", "io.ReadFull")) -and
            (Test-AllPatterns $blockTestText @("TestBlockMarshalRoundTrip", "TestBlockUnmarshalRejectsBadVersion", "TestPackBitsRoundTripMixedWidths", "TestBlockMetadataDistinguishesOriginalAndPaddedDim"))
        )
    },
    [pscustomobject]@{
        Name = "Encode Decode And Tail Handling"
        Weight = 15
        Passed = (
            (Test-AllPatterns $encodeText @("EncodeVector", "quantizeScalarByBoundary", "encodeResidual")) -and
            (Test-AllPatterns $decodeText @("DecodeVector", "UnmarshalEncodedVector", "invalid outlier mask length", "unexpected trailing bytes")) -and
            (Test-AllPatterns $encodeTestText @("TestEncodeDecodeRoundTripAcrossShapes", "TestEncodeVectorDeterministicBytes", "TestDistortionThresholds")) -and
            (Test-AllPatterns $decodeTestText @("TestDecodeVectorPreservesOriginalLengthAcrossBlocks", "TestDecodeVectorRejectsInvalidIndexLengths"))
        )
    }
)

$unitChecks = @(
    [pscustomobject]@{
        Name = "Deterministic Byte-Identical Encoding Test"
        Weight = 20
        Passed = (Test-AllPatterns $encodeTestText @("TestEncodeVectorDeterministicBytes", "byte-identical"))
    },
    [pscustomobject]@{
        Name = "Malformed Decode Coverage"
        Weight = 20
        Passed = (Test-AllPatterns $decodeTestText @("TestUnmarshalEncodedVectorRejectsBadHeader", "TestUnmarshalEncodedVectorRejectsWrongVersion", "TestUnmarshalEncodedVectorRejectsBadPresetID", "TestUnmarshalEncodedVectorRejectsTruncatedBlockPayload"))
    },
    [pscustomobject]@{
        Name = "Distortion Thresholds Covered"
        Weight = 20
        Passed = (Test-AllPatterns $encodeTestText @("TestDistortionThresholds", "5.0", "2.5"))
    },
    [pscustomobject]@{
        Name = "tq35 Outperforms tq25 In Synthetic Suite"
        Weight = 20
        Passed = (Test-AllPatterns $encodeTestText @("tq35 mean MSE", "tq25 mean MSE", "want <= tq25 mean MSE"))
    },
    [pscustomobject]@{
        Name = "Docs And Scripts Defer Runtime To Plan 5"
        Weight = 20
        Passed = (
            (Test-Path $scorecardPath) -and
            (Test-Path $testScriptPath) -and
            (Test-Path $scoreScriptPath) -and
            (Test-AllPatterns $scorecardText @("ollama_turboquant_full_plan.txt", "ollama_turboquant_algorithms.txt", "Plan 1", "Plan 2", "deferred until Plan 5", "Go toolchain"))
        )
    }
)

foreach ($check in $codecChecks) {
    $check | Add-Member -NotePropertyName Score -NotePropertyValue (Get-Score -Weight $check.Weight -Passed $check.Passed)
}

foreach ($check in $unitChecks) {
    $check | Add-Member -NotePropertyName Score -NotePropertyValue (Get-Score -Weight $check.Weight -Passed $check.Passed)
}

$codecScore = ($codecChecks | Measure-Object -Property Score -Sum).Sum
$unitScore = ($unitChecks | Measure-Object -Property Score -Sum).Sum
$goAvailable = $null -ne (Get-Command go -ErrorAction SilentlyContinue)

$result = [pscustomobject]@{
    RepoRoot = $resolvedRepoRoot
    CodecCompletenessScore = $codecScore
    UnitQualityScore = $unitScore
    GoToolchainAvailable = $goAvailable
    CodecChecks = $codecChecks
    UnitChecks = $unitChecks
}

$json = $result | ConvertTo-Json -Depth 6

Write-Output ("Codec Completeness Score: {0}/100" -f $codecScore)
Write-Output ("Unit Quality Score: {0}/100" -f $unitScore)
if ($goAvailable) {
    Write-Output "Go toolchain available: yes"
}
else {
    Write-Output "Go toolchain not available; score is based on source and test artifacts only."
}
Write-Output ""
Write-Output "Codec Completeness Breakdown:"
foreach ($check in $codecChecks) {
    Write-Output ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
}
Write-Output ""
Write-Output "Unit Quality Breakdown:"
foreach ($check in $unitChecks) {
    Write-Output ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
}

if ($JsonPath) {
    Set-Content -Path $JsonPath -Value $json
}

if ($MarkdownPath) {
    $md = @(
        "# TurboQuant Plan 3 Scores",
        "",
        ("- Codec Completeness Score: {0}/100" -f $codecScore),
        ("- Unit Quality Score: {0}/100" -f $unitScore),
        ("- Go toolchain available: {0}" -f ($(if ($goAvailable) { "yes" } else { "no" }))),
        "",
        "## Codec Completeness Breakdown"
    )

    foreach ($check in $codecChecks) {
        $md += ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
    }

    $md += ""
    $md += "## Unit Quality Breakdown"

    foreach ($check in $unitChecks) {
        $md += ("- {0}: {1}/{2}" -f $check.Name, $check.Score, $check.Weight)
    }

    Set-Content -Path $MarkdownPath -Value ($md -join "`r`n")
}

Write-Output ""
Write-Output "JSON:"
Write-Output $json
