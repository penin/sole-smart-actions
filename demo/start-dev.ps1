# Development: backend in a new window + Vite dev server here.
. (Join-Path $PSScriptRoot "_common.ps1")
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $DemoDir "start-backend.ps1")
& (Join-Path $DemoDir "start-frontend.ps1")
