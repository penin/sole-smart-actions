# Shared helpers for the demo start scripts.
$ErrorActionPreference = "Stop"
$DemoDir = $PSScriptRoot
$ProjectRoot = Split-Path $DemoDir -Parent
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Assert-Python {
    if (-not (Test-Path $Python)) {
        throw "Project virtual environment not found at $Python. See demo\README.md (Installation)."
    }
    & $Python -c "import fastapi, uvicorn, sole_intent" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing demo backend dependencies into the project .venv..." -ForegroundColor Yellow
        & $Python -m pip install -r (Join-Path $DemoDir "backend\requirements.txt")
        & $Python -m pip install -e $ProjectRoot
    }
}

function Find-Npm {
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($npm) { return $npm.Source }
    $portable = Get-ChildItem "$env:LOCALAPPDATA\Programs\node-portable\node-v*-win-x64\npm.cmd" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
    if ($portable) {
        $env:Path = "$($portable.DirectoryName);$env:Path"
        return $portable.FullName
    }
    return $null
}
