# Presentation mode: one process serves the API and the built UI at http://127.0.0.1:8000
param([int]$Port = 8000, [switch]$Rebuild, [switch]$NoBrowser)
. (Join-Path $PSScriptRoot "_common.ps1")
Assert-Python

$dist = Join-Path $DemoDir "frontend\dist\index.html"
if ($Rebuild -or -not (Test-Path $dist)) {
    $npm = Find-Npm
    if (-not $npm) { throw "Frontend is not built and Node.js/npm was not found. See demo\README.md." }
    Push-Location (Join-Path $DemoDir "frontend")
    try {
        if (-not (Test-Path "node_modules")) { & $npm ci }
        & $npm run build
        if ($LASTEXITCODE -ne 0) { throw "Frontend build failed" }
    } finally { Pop-Location }
}

$url = "http://127.0.0.1:$Port"
Write-Host "`nSole Smart Actions demo -> $url   (Ctrl+C to stop)`n" -ForegroundColor Cyan
if (-not $NoBrowser) {
    Start-Job -ScriptBlock { param($u) Start-Sleep -Seconds 3; Start-Process $u } -ArgumentList $url | Out-Null
}
& $Python (Join-Path $DemoDir "backend\main.py") --port $Port
