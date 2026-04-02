param(
    [switch]$IncludePlatformGated
)

$ErrorActionPreference = "Stop"
$global:PSNativeCommandUseErrorActionPreference = $false

$repoRoot = Split-Path -Parent $PSScriptRoot
$go = "C:\Program Files\Go\bin\go.exe"
if (-not (Test-Path $go)) {
    $go = "go"
}

$laneA = @(
    "./api",
    "./cmd/bench",
    "./cmd/benchkv",
    "./turboquant"
)

$laneB = @(
    "./cmd",
    "./runner/ollamarunner",
    "./llm",
    "./ml/backend/ggml"
)

$laneC = @("x/mlxrunner/mlx", "x/imagegen/...")

$passed = @()
$failed = @()
$skipped = @()

foreach ($pkg in $laneA) {
    Push-Location $repoRoot
    try {
        $command = '"' + $go + '" test ' + $pkg + ' 2>&1'
        $output = cmd /c $command
        if ($LASTEXITCODE -eq 0) {
            $passed += @{ Package = $pkg; Output = $output }
        } else {
            $failed += @{ Package = $pkg; Output = $output }
        }
    } finally {
        Pop-Location
    }
}

if ($IncludePlatformGated) {
    foreach ($pkg in $laneB) {
        Push-Location $repoRoot
        try {
            $command = '"' + $go + '" test ' + $pkg + ' 2>&1'
            $output = cmd /c $command
            if ($LASTEXITCODE -eq 0) {
                $passed += @{ Package = $pkg; Output = $output }
            } else {
                $failed += @{ Package = $pkg; Output = $output }
            }
        } finally {
            Pop-Location
        }
    }
} else {
    foreach ($pkg in $laneB) {
        $skipped += @{ Package = $pkg; Reason = "platform/build-tag gated; rerun with -IncludePlatformGated" }
    }
}

foreach ($pkg in $laneC) {
    $skipped += @{ Package = $pkg; Reason = "excluded local subsystem lane on this machine" }
}

Write-Host "TurboQuant local test lanes"
Write-Host ""

Write-Host "Passed:"
if ($passed.Count -eq 0) {
    Write-Host "  (none)"
} else {
    foreach ($item in $passed) {
        Write-Host "  $($item.Package)"
    }
}

Write-Host ""
Write-Host "Failed:"
if ($failed.Count -eq 0) {
    Write-Host "  (none)"
} else {
    foreach ($item in $failed) {
        Write-Host "  $($item.Package)"
        $item.Output | ForEach-Object { Write-Host "    $_" }
    }
}

Write-Host ""
Write-Host "Skipped:"
if ($skipped.Count -eq 0) {
    Write-Host "  (none)"
} else {
    foreach ($item in $skipped) {
        Write-Host "  $($item.Package) - $($item.Reason)"
    }
}

if ($failed.Count -gt 0) {
    exit 1
}
