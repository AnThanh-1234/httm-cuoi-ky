$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$log = Join-Path $root 'progress.log'
$owner = 'hoaiphan1978'
$train = @('titans-vi-memory-stable', 'ttt-linear-vi-memory-parallel')
$compare = 'compare-titans-vs-ttt-parallel'
$kaggle = (Get-Command kaggle -CommandType Application).Source
$deadline = (Get-Date).AddHours(10)
function Write-ProgressLog([string] $message) {
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format s) $message"
}
function Wait-Kernels($slugs) {
    while ((Get-Date) -lt $deadline) {
        $completed = 0
        foreach ($slug in $slugs) {
            $status = (& $kaggle kernels status "$owner/$slug" 2>&1 | Out-String).Trim()
            Write-ProgressLog $status
            if ($LASTEXITCODE -ne 0) { continue }
            if ($status -match 'KernelWorkerStatus\.(ERROR|CANCEL|FAILED)') {
                & $kaggle kernels logs "$owner/$slug" 2>&1 |
                    Set-Content -LiteralPath (Join-Path $root "$slug-error.log")
                throw "$slug failed"
            }
            if ($status -match 'KernelWorkerStatus\.COMPLETE') { $completed++ }
        }
        if ($completed -eq $slugs.Count) { return }
        Start-Sleep -Seconds 45
    }
    throw 'Polling deadline elapsed'
}
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $envFile = Join-Path (Split-Path $root) '.env'
    $line = Get-Content -LiteralPath $envFile | Where-Object { $_ -match '^\s*(?:export\s+)?kaggle_key\s*=' } | Select-Object -First 1
    if (-not $line) { throw 'kaggle_key missing' }
    $env:KAGGLE_API_TOKEN = ($line -split '=',2)[1].Trim().Trim('"').Trim("'")
    Wait-Kernels $train
    for ($i=0; $i -lt $train.Count; $i++) {
        $files = (& $kaggle kernels files "$owner/$($train[$i])" 2>&1 | Out-String).Trim()
        Write-ProgressLog $files
        $required = @('titans_vi_memory.pt','ttt_linear_vi_memory.pt')[$i]
        if ($LASTEXITCODE -ne 0 -or -not $files.Contains($required)) { throw "Missing checkpoint $required" }
    }
    $push = (& $kaggle kernels push -p (Join-Path $root $compare) --accelerator NvidiaTeslaT4 2>&1 | Out-String).Trim()
    Write-ProgressLog $push
    if ($LASTEXITCODE -ne 0 -or $push -notmatch 'successfully pushed' -or $push -match 'not valid kernel sources') {
        throw 'Could not launch comparison with both training outputs'
    }
    Wait-Kernels @($compare)
    $files = (& $kaggle kernels files "$owner/$compare" 2>&1 | Out-String).Trim()
    Write-ProgressLog $files
    if ($LASTEXITCODE -ne 0 -or -not $files.Contains('compare_results.json')) { throw 'Comparison results missing' }
} catch {
    Write-ProgressLog $_.ToString()
    exit 1
}

