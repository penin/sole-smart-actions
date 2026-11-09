# Runs backend (pytest) and frontend (vitest + type-check) test suites.
. (Join-Path $PSScriptRoot "_common.ps1")
Assert-Python
Push-Location (Join-Path $DemoDir "backend")
try { & $Python -m pytest -q; $backend = $LASTEXITCODE } finally { Pop-Location }
$npm = Find-Npm
Push-Location (Join-Path $DemoDir "frontend")
try { & $npm run typecheck; $tc = $LASTEXITCODE; & $npm test; $frontend = $LASTEXITCODE } finally { Pop-Location }
Write-Host "`nbackend: $(if ($backend -eq 0) {'PASS'} else {'FAIL'})  typecheck: $(if ($tc -eq 0) {'PASS'} else {'FAIL'})  frontend: $(if ($frontend -eq 0) {'PASS'} else {'FAIL'})"
if ($backend -or $tc -or $frontend) { exit 1 }
