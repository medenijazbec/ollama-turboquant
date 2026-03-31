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

$apiTypesText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "api\types.go")
$apiTypesTestText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "api\types_test.go")
$routesOptionsText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "server\routes_options_test.go")
$llmServerText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "llm\server.go")
$llmServerTestText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "llm\server_test.go")
$cmdText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "cmd\cmd.go")
$cmdTestText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "cmd\cmd_test.go")
$benchText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "cmd\bench\bench.go")
$benchTestText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "cmd\bench\bench_test.go")
$turboquantDocText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "docs\turboquant.mdx")
$cliDocText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "docs\cli.mdx")
$apiDocText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "docs\api.md")
$modelfileDocText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "docs\modelfile.mdx")
$docsNavText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "docs\docs.json")
$openAPIText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "docs\openapi.yaml")
$matrixText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "docs\turboquant_completion_matrix.md")
$scorecardText = Read-FileOrEmpty (Join-Path $resolvedRepoRoot "docs\turboquant_completion_scorecard.md")

$implementationChecks = @(
    [pscustomobject]@{
        Name = "API and model option plumbing exists"
        Weight = 20
        Passed = (
            (Test-AllPatterns $apiTypesText @("KVCacheType string")) -and
            (Test-AllPatterns $apiTypesTestText @("TestKVCacheTypeParsingFromJSON", "TestKVCacheTypeFormatParams")) -and
            (Test-AllPatterns $routesOptionsText @("TestModelOptionsKVCacheTypePriority"))
        )
    },
    [pscustomobject]@{
        Name = "llm server resolves request over env"
        Weight = 20
        Passed = (
            (Test-AllPatterns $llmServerText @("resolveKVCacheMode", "opts.KVCacheType", "envconfig.KvCacheType()")) -and
            (Test-AllPatterns $llmServerTestText @("TestResolveKVCacheMode"))
        )
    },
    [pscustomobject]@{
        Name = "ollama run exposes --turboquant"
        Weight = 20
        Passed = (
            (Test-AllPatterns $cmdText @("normalizeTurboQuantFlag", 'Lookup("turboquant").NoOptDefVal = "tq35"', 'opts.Options["kv_cache_type"]')) -and
            (Test-AllPatterns $cmdTestText @("TestRunHandlerTurboQuantAddsGenerateOption", "TestLoadOrUnloadModelIncludesKVCacheTypeOption"))
        )
    },
    [pscustomobject]@{
        Name = "ollama-bench exposes -turboquant"
        Weight = 20
        Passed = (
            (Test-AllPatterns $benchText @('flag.String("turboquant"', 'options["kv_cache_type"]', "| KV:", "| Path:")) -and
            (Test-AllPatterns $benchTestText @("TestBuildGenerateRequestIncludesKVCacheType"))
        )
    },
    [pscustomobject]@{
        Name = "Completion audit artifacts exist"
        Weight = 20
        Passed = (
            (Test-AllPatterns $matrixText @("TurboQuant Completion Matrix", "Productization")) -and
            (Test-AllPatterns $scorecardText @("Implementation Completeness Score", "Product Surface Completeness Score"))
        )
    }
)

$productChecks = @(
    [pscustomobject]@{
        Name = "CLI docs cover TurboQuant"
        Weight = 20
        Passed = (Test-AllPatterns $cliDocText @("--turboquant", "--turboquant=tq25", "--turboquant=off"))
    },
    [pscustomobject]@{
        Name = "API docs and schema cover kv_cache_type"
        Weight = 20
        Passed = (
            (Test-AllPatterns $apiDocText @("kv_cache_type", "tq35")) -and
            (Test-AllPatterns $openAPIText @("kv_cache_type:"))
        )
    },
    [pscustomobject]@{
        Name = "Modelfile docs cover selected-model defaults"
        Weight = 20
        Passed = (Test-AllPatterns $modelfileDocText @("PARAMETER kv_cache_type tq35", "PARAMETER kv_cache_type tq25"))
    },
    [pscustomobject]@{
        Name = "Dedicated TurboQuant guide exists and is linked"
        Weight = 20
        Passed = (
            (Test-AllPatterns $turboquantDocText @("TurboQuant is KV-cache compression only", "Results Template", "GGUF")) -and
            (Test-AllPatterns $docsNavText @('"/turboquant"'))
        )
    },
    [pscustomobject]@{
        Name = "Completion scorecard describes pre/post outcomes"
        Weight = 20
        Passed = (Test-AllPatterns $scorecardText @("85 -> 100", "40 -> 100"))
    }
)

foreach ($check in $implementationChecks) {
    $check | Add-Member -NotePropertyName Score -NotePropertyValue (Get-Score -Weight $check.Weight -Passed $check.Passed)
}

foreach ($check in $productChecks) {
    $check | Add-Member -NotePropertyName Score -NotePropertyValue (Get-Score -Weight $check.Weight -Passed $check.Passed)
}

$implementationScore = ($implementationChecks | Measure-Object -Property Score -Sum).Sum
$productScore = ($productChecks | Measure-Object -Property Score -Sum).Sum
$goAvailable = $null -ne (Get-Command go -ErrorAction SilentlyContinue)

$result = [pscustomobject]@{
    RepoRoot = $resolvedRepoRoot
    ImplementationCompletenessScore = $implementationScore
    ProductSurfaceCompletenessScore = $productScore
    GoToolchainAvailable = $goAvailable
    ImplementationChecks = $implementationChecks
    ProductChecks = $productChecks
}

$json = $result | ConvertTo-Json -Depth 6

Write-Output ("Implementation Completeness Score: {0}/100" -f $implementationScore)
Write-Output ("Product Surface Completeness Score: {0}/100" -f $productScore)
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
        "# TurboQuant Completion Scores",
        "",
        ("- Implementation Completeness Score: {0}/100" -f $implementationScore),
        ("- Product Surface Completeness Score: {0}/100" -f $productScore),
        ("- Go toolchain available: {0}" -f ($(if ($goAvailable) { "yes" } else { "no" })))
    )
    Set-Content -Path $MarkdownPath -Value ($md -join "`r`n")
}

Write-Output ""
Write-Output "JSON:"
Write-Output $json
