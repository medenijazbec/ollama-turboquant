param(
    [string]$AuditPath = (Join-Path $PSScriptRoot "..\docs\turboquant_audit.md"),
    [string]$RepoRoot = (Join-Path $PSScriptRoot "..")
)

$ErrorActionPreference = "Stop"

$resolvedAuditPath = (Resolve-Path $AuditPath).Path
$resolvedRepoRoot = (Resolve-Path $RepoRoot).Path
$text = Get-Content -Raw $resolvedAuditPath

$requiredSections = @(
    "Source Basis",
    "Cross-Reference Matrix",
    "OLLAMA_KV_CACHE_TYPE Trace",
    "Hard-Coded Cache-Type Assumptions",
    "KV Memory Estimation",
    "Cache Contract And Tensor Shapes",
    "Attention And Backend Insertion Points",
    "Legacy Runner Handling",
    "Plan 2-5 Insertion Points",
    "Gap Analysis"
)

$requiredSymbols = @(
    "KvCacheType",
    "SupportsKVCacheType",
    "KVCacheTypeIsQuantized",
    "GraphSize",
    "cache.Init",
    "NewInputCache",
    "Get",
    "Put",
    "ScaledDotProductAttention",
    "NewContextParams"
)

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

$requiredTxtSources = @(
    "ollama_turboquant_full_plan.txt",
    "ollama_turboquant_algorithms.txt"
)

$errors = New-Object System.Collections.Generic.List[string]

foreach ($section in $requiredSections) {
    if ($text -notmatch [regex]::Escape($section)) {
        $errors.Add("Missing required section: $section")
    }
}

foreach ($symbol in $requiredSymbols) {
    if ($text -notmatch [regex]::Escape($symbol)) {
        $errors.Add("Missing required symbol mention: $symbol")
    }
}

foreach ($file in $requiredFiles) {
    if ($text -notmatch [regex]::Escape($file)) {
        $errors.Add("Missing required file reference in audit: $file")
    }

    $fullPath = Join-Path $resolvedRepoRoot $file
    if (-not (Test-Path $fullPath)) {
        $errors.Add("Expected repo file does not exist: $file")
    }
}

foreach ($txtSource in $requiredTxtSources) {
    if ($text -notmatch [regex]::Escape($txtSource)) {
        $errors.Add("Missing required TurboQuant source citation: $txtSource")
    }
}

if ($text -notmatch "Plan 2-5 insertion points" -and $text -notmatch "Plan 2" -and $text -notmatch "Plans 2-5") {
    $errors.Add("Missing Plan 2-5 handoff section")
}

if ($text -notmatch "Phase A" -or $text -notmatch "Phase B") {
    $errors.Add("Audit must explicitly distinguish Phase A and Phase B")
}

if ($text -match "(?i)(?<!not a claim that )(?:full TurboQuant is already implemented|fully implemented today)") {
    $errors.Add("Audit incorrectly claims full TurboQuant implementation status")
}

if ($text -notmatch "not a claim that full TurboQuant is already implemented today" -and $text -notmatch "Full TurboQuant should only be claimed once Phase B exists and is validated") {
    $errors.Add("Audit must explicitly state that full TurboQuant is deferred beyond Plan 1")
}

if ($errors.Count -gt 0) {
    Write-Error ("TurboQuant audit validation failed:`n- " + ($errors -join "`n- "))
    exit 1
}

Write-Output "TurboQuant audit validation passed."
