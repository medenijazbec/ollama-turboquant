param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot "..")
)

$ErrorActionPreference = "Stop"

$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path

$checks = @(
    @{ Path = "turboquant\turboquant.go"; Patterns = @("RegularCodebook", "RegularBoundaries", "OutlierCodebook", "OutlierBoundaries", 'case "tq3", "tq4":', 'return "tq35"') },
    @{ Path = "turboquant\rotation.go"; Patterns = @("BuildRotation", "ApplyRotation", "ApplyInverseRotation") },
    @{ Path = "turboquant\codebook.go"; Patterns = @("explicitCodebook", "codebookBoundaries", "quantizeScalarByBoundary", "selectOutliers") },
    @{ Path = "turboquant\residual_qjl.go"; Patterns = @("encodeResidual", "reconstructResidual", "residualDotCorrection") },
    @{ Path = "turboquant\block.go"; Patterns = @("MarshalBinary", "UnmarshalBinary", "expectedPackedBytes", "io.ReadFull") },
    @{ Path = "turboquant\encode.go"; Patterns = @("EncodeVector", "quantizeScalarByBoundary", "RegularBoundaries", "OutlierBoundaries") },
    @{ Path = "turboquant\decode.go"; Patterns = @("DecodeVector", "UnmarshalEncodedVector", "unexpected trailing bytes", "invalid outlier mask length") },
    @{ Path = "turboquant\stats.go"; Patterns = @("RMSE", "MeanAbsErr", "MaxAbsErr") },
    @{ Path = "turboquant\rotation_test.go"; Patterns = @("TestBuildRotationDeterministic", "TestBuildRotationDifferentSeedsDiffer", "TestApplyInverseRotation", "TestRotationPreservesNorm") },
    @{ Path = "turboquant\codebook_test.go"; Patterns = @("TestPresetCodebookLengths", "TestCodebookBoundariesMonotonic", "TestQuantizeScalarByBoundaryDeterministic", "TestSelectOutliersStableOnTies") },
    @{ Path = "turboquant\residual_qjl_test.go"; Patterns = @("TestResidualSketchDeterministic", "TestReconstructResidualLength", "TestZeroResidualProducesZeroReconstruction", "TestResidualDotCorrectionDeterministicAndFinite") },
    @{ Path = "turboquant\block_test.go"; Patterns = @("TestBlockMarshalRoundTrip", "TestBlockUnmarshalRejectsBadVersion", "TestPackBitsRoundTripMixedWidths", "TestBlockMetadataDistinguishesOriginalAndPaddedDim") },
    @{ Path = "turboquant\encode_test.go"; Patterns = @("TestEncodeDecodeRoundTripAcrossShapes", "TestEncodeVectorDeterministicBytes", "TestPresetAliases", "TestDistortionThresholds", "5.0", "2.5") },
    @{ Path = "turboquant\decode_test.go"; Patterns = @("TestUnmarshalEncodedVectorRejectsBadHeader", "TestUnmarshalEncodedVectorRejectsWrongVersion", "TestUnmarshalEncodedVectorRejectsBadPresetID", "TestUnmarshalEncodedVectorRejectsTruncatedBlockPayload", "TestDecodeVectorRejectsInvalidIndexLengths") },
    @{ Path = "turboquant\stats_test.go"; Patterns = @("TestCompareIdenticalVectors", "TestCompareKnownExample", "TestCompareRMSEMatchesSqrtMSE") },
    @{ Path = "docs\turboquant_plan3_scorecard.md"; Patterns = @("ollama_turboquant_full_plan.txt", "ollama_turboquant_algorithms.txt", "Plan 1", "Plan 2", "deferred until Plan 5", "Codec Completeness Score", "Unit Quality Score", "pre", "post") },
    @{ Path = "scripts\score_turboquant_plan3.ps1"; Patterns = @("Codec Completeness Score", "Unit Quality Score", "Go toolchain") }
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
    Write-Error ("TurboQuant Plan 3 validation failed:`n- " + ($errors -join "`n- "))
    exit 1
}

$go = Get-Command go -ErrorAction SilentlyContinue
if ($null -ne $go) {
    Push-Location $resolvedRepoRoot
    try {
        & go test ./turboquant
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Output "Go toolchain not available; skipped 'go test ./turboquant' execution."
}

Write-Output "TurboQuant Plan 3 validation passed."
