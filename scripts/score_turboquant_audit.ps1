param(
    [string]$AuditPath = (Join-Path $PSScriptRoot "..\docs\turboquant_audit.md"),
    [string]$RepoRoot = (Join-Path $PSScriptRoot ".."),
    [string]$JsonPath,
    [string]$MarkdownPath
)

$ErrorActionPreference = "Stop"

function Test-ContainsAll {
    param(
        [string]$Text,
        [string[]]$Needles
    )

    foreach ($needle in $Needles) {
        if ($Text -notmatch [regex]::Escape($needle)) {
            return $false
        }
    }

    return $true
}

function Test-ContainsAny {
    param(
        [string]$Text,
        [string[]]$Needles
    )

    foreach ($needle in $Needles) {
        if ($Text -match [regex]::Escape($needle)) {
            return $true
        }
    }

    return $false
}

function Get-ScaledScore {
    param(
        [int]$Points,
        [int]$Passed,
        [int]$Total
    )

    if ($Total -le 0) {
        return 0
    }

    return [int][math]::Round($Points * ($Passed / $Total), 0, [MidpointRounding]::AwayFromZero)
}

function New-CheckResult {
    param(
        [string]$Name,
        [int]$Weight,
        [string[]]$Checks,
        [string]$Text
    )

    $passed = 0
    $missing = @()

    foreach ($check in $Checks) {
        if ($Text -match [regex]::Escape($check)) {
            $passed++
        } else {
            $missing += $check
        }
    }

    [pscustomobject]@{
        Name = $Name
        Weight = $Weight
        Passed = $passed
        Total = $Checks.Count
        Score = Get-ScaledScore -Points $Weight -Passed $passed -Total $Checks.Count
        Missing = $missing
    }
}

$resolvedAuditPath = (Resolve-Path $AuditPath).Path
$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path
$text = Get-Content -Raw $resolvedAuditPath

$requiredFiles = @(
    "envconfig/config.go",
    "llm/server.go",
    "fs/ggml/ggml.go",
    "runner/ollamarunner/cache.go",
    "runner/ollamarunner/runner.go",
    "kvcache/cache.go",
    "kvcache/causal.go",
    "ml/backend.go",
    "ml/nn/attention.go",
    "ml/backend/ggml/ggml.go",
    "ml/backend/ggml/ggml/src/ggml-backend.cpp",
    "llama/llama.go"
)

$requiredSymbols = @(
    "KvCacheType",
    "SupportsKVCacheType",
    "KVCacheTypeIsQuantized",
    "GraphSize",
    "kvCacheBytesPerElement",
    "cache.Init",
    "NewInputCache",
    "Get",
    "Put",
    "CopyPrefix",
    "CanResume",
    "Remove",
    "ScaledDotProductAttention",
    "NewContextParams"
)

$futureFastPathFiles = @(
    "ml/backend/ggml/ggml/src/ggml-backend.cpp",
    "ml/backend/ggml/ggml/src/ggml.c",
    "ml/backend/ggml/ggml/src/ggml-quants.c"
)

$auditCompletenessChecks = @(
    (New-CheckResult -Name "Source Basis And Cross Map" -Weight 15 -Checks @(
        "ollama_turboquant_full_plan.txt",
        "ollama_turboquant_algorithms.txt",
        "cross-reference matrix"
    ) -Text $text),
    (New-CheckResult -Name "KV Cache Type Trace" -Weight 15 -Checks @(
        "OLLAMA_KV_CACHE_TYPE",
        "envconfig/config.go",
        "llm/server.go",
        "runner/ollamarunner/cache.go",
        "runner/ollamarunner/runner.go",
        "kvcache/cache.go",
        "kvcache/causal.go",
        "ml/nn/attention.go",
        "ml/backend/ggml/ggml.go"
    ) -Text $text),
    (New-CheckResult -Name "Hard-Coded Cache Assumptions" -Weight 15 -Checks @(
        "f16",
        "q8_0",
        "q4_0",
        "hard-coded",
        "assumption"
    ) -Text $text),
    (New-CheckResult -Name "KV Memory Estimation" -Weight 10 -Checks @(
        "GraphSize",
        "kvCacheBytesPerElement",
        "llm/server.go",
        "fs/ggml/ggml.go"
    ) -Text $text),
    (New-CheckResult -Name "Cache Contract And Shapes" -Weight 10 -Checks @(
        "Get",
        "Put",
        "CopyPrefix",
        "CanResume",
        "Remove",
        "[head_dim, kv_heads, batch]",
        "[history, batch]"
    ) -Text $text),
    (New-CheckResult -Name "Attention Compute Path" -Weight 10 -Checks @(
        "ScaledDotProductAttention",
        "ml/nn/attention.go",
        "flash attention",
        "fallback"
    ) -Text $text),
    (New-CheckResult -Name "Legacy Runner Handling" -Weight 10 -Checks @(
        "llama/llama.go",
        "NewContextParams",
        "legacy",
        "reject"
    ) -Text $text),
    (New-CheckResult -Name "Plan 2-5 Insertion Points" -Weight 15 -Checks @(
        "Plan 2",
        "Plan 3",
        "Plan 4",
        "Plan 5",
        "ggml-backend.cpp",
        "ggml.c",
        "ggml-quants.c"
    ) -Text $text)
)

$existingFileCount = 0
$mentionedFileCount = 0
$missingRequiredFiles = @()
foreach ($file in $requiredFiles) {
    $fullPath = Join-Path $resolvedRepoRoot $file
    if (Test-Path $fullPath) {
        $existingFileCount++
    }
    if ($text -match [regex]::Escape($file)) {
        $mentionedFileCount++
    } else {
        $missingRequiredFiles += $file
    }
}

$existingFutureFastPathCount = 0
$mentionedFutureFastPathCount = 0
$missingFutureFastPathFiles = @()
foreach ($file in $futureFastPathFiles) {
    $fullPath = Join-Path $resolvedRepoRoot $file
    if (Test-Path $fullPath) {
        $existingFutureFastPathCount++
    }
    if ($text -match [regex]::Escape($file)) {
        $mentionedFutureFastPathCount++
    } else {
        $missingFutureFastPathFiles += $file
    }
}

$mentionedSymbolCount = 0
$missingSymbols = @()
foreach ($symbol in $requiredSymbols) {
    if ($text -match [regex]::Escape($symbol)) {
        $mentionedSymbolCount++
    } else {
        $missingSymbols += $symbol
    }
}

$repoSeamChecks = @(
    "normalizeKVCacheType",
    "kvCacheTypeFromStr",
    "cache.Init",
    "NewInputCache",
    "GraphSize",
    "ScaledDotProductAttention",
    "ggml_backend_graph_compute_async",
    "ggml_backend_tensor_set_async"
)
$seamCount = 0
$missingSeams = @()
foreach ($seam in $repoSeamChecks) {
    if ($text -match [regex]::Escape($seam)) {
        $seamCount++
    } else {
        $missingSeams += $seam
    }
}

$auditReferenceTargets = $requiredFiles + $futureFastPathFiles
$resolvedReferenceCount = 0
$missingResolvedReferences = @()
foreach ($target in ($auditReferenceTargets | Select-Object -Unique)) {
    if ($text -match [regex]::Escape($target)) {
        $fullPath = Join-Path $resolvedRepoRoot $target
        if (Test-Path $fullPath) {
            $resolvedReferenceCount++
        } else {
            $missingResolvedReferences += $target
        }
    }
}

$staticCoverageChecks = @(
    [pscustomobject]@{
        Name = "Required Files Exist And Are Referenced"
        Weight = 20
        Passed = $mentionedFileCount
        Total = $requiredFiles.Count
        Score = Get-ScaledScore -Points 20 -Passed $mentionedFileCount -Total $requiredFiles.Count
        Missing = $missingRequiredFiles
    },
    [pscustomobject]@{
        Name = "Required Symbols Mentioned"
        Weight = 20
        Passed = $mentionedSymbolCount
        Total = $requiredSymbols.Count
        Score = Get-ScaledScore -Points 20 -Passed $mentionedSymbolCount -Total $requiredSymbols.Count
        Missing = $missingSymbols
    },
    [pscustomobject]@{
        Name = "Required Repo Seams Discovered"
        Weight = 20
        Passed = $seamCount
        Total = $repoSeamChecks.Count
        Score = Get-ScaledScore -Points 20 -Passed $seamCount -Total $repoSeamChecks.Count
        Missing = $missingSeams
    },
    [pscustomobject]@{
        Name = "Future Fast-Path Targets Named"
        Weight = 20
        Passed = $mentionedFutureFastPathCount
        Total = $futureFastPathFiles.Count
        Score = Get-ScaledScore -Points 20 -Passed $mentionedFutureFastPathCount -Total $futureFastPathFiles.Count
        Missing = $missingFutureFastPathFiles
    },
    [pscustomobject]@{
        Name = "Audit-To-Code References Resolve"
        Weight = 20
        Passed = $resolvedReferenceCount
        Total = ($auditReferenceTargets | Select-Object -Unique).Count
        Score = Get-ScaledScore -Points 20 -Passed $resolvedReferenceCount -Total (($auditReferenceTargets | Select-Object -Unique).Count)
        Missing = $missingResolvedReferences
    }
)

$auditCompletenessScore = ($auditCompletenessChecks | Measure-Object -Property Score -Sum).Sum
$staticCoverageScore = ($staticCoverageChecks | Measure-Object -Property Score -Sum).Sum

$result = [pscustomobject]@{
    AuditPath = $resolvedAuditPath
    RepoRoot = $resolvedRepoRoot
    AuditCompletenessScore = $auditCompletenessScore
    StaticPipelineCoverageScore = $staticCoverageScore
    AuditCompletenessChecks = $auditCompletenessChecks
    StaticPipelineCoverageChecks = $staticCoverageChecks
    ExistingRequiredFileCount = $existingFileCount
    ExistingFutureFastPathFileCount = $existingFutureFastPathCount
}

$json = $result | ConvertTo-Json -Depth 6

Write-Output ("Audit Completeness Score: {0}/100" -f $auditCompletenessScore)
Write-Output ("Static Pipeline Coverage Score: {0}/100" -f $staticCoverageScore)
Write-Output ""
Write-Output "Audit Completeness Breakdown:"
foreach ($check in $auditCompletenessChecks) {
    Write-Output ("- {0}: {1}/{2} => {3}/{4}" -f $check.Name, $check.Passed, $check.Total, $check.Score, $check.Weight)
}
Write-Output ""
Write-Output "Static Pipeline Coverage Breakdown:"
foreach ($check in $staticCoverageChecks) {
    Write-Output ("- {0}: {1}/{2} => {3}/{4}" -f $check.Name, $check.Passed, $check.Total, $check.Score, $check.Weight)
}

if ($JsonPath) {
    Set-Content -Path $JsonPath -Value $json
}

if ($MarkdownPath) {
    $md = @(
        "# TurboQuant Audit Scores",
        "",
        ("- Audit Completeness Score: {0}/100" -f $auditCompletenessScore),
        ("- Static Pipeline Coverage Score: {0}/100" -f $staticCoverageScore),
        "",
        "## Audit Completeness Breakdown"
    )

    foreach ($check in $auditCompletenessChecks) {
        $md += ("- {0}: {1}/{2} => {3}/{4}" -f $check.Name, $check.Passed, $check.Total, $check.Score, $check.Weight)
    }

    $md += ""
    $md += "## Static Pipeline Coverage Breakdown"

    foreach ($check in $staticCoverageChecks) {
        $md += ("- {0}: {1}/{2} => {3}/{4}" -f $check.Name, $check.Passed, $check.Total, $check.Score, $check.Weight)
    }

    Set-Content -Path $MarkdownPath -Value ($md -join "`r`n")
}

Write-Output ""
Write-Output "JSON:"
Write-Output $json
