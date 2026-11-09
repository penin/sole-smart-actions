# Vite dev server with hot reload at http://127.0.0.1:5173 (proxies /api to the backend on :8000)
. (Join-Path $PSScriptRoot "_common.ps1")
$npm = Find-Npm
if (-not $npm) { throw "Node.js/npm not found. See demo\README.md." }
Push-Location (Join-Path $DemoDir "frontend")
try {
    if (-not (Test-Path "node_modules")) { & $npm ci }
    & $npm run dev
} finally { Pop-Location }
