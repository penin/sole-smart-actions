# API only (development). http://127.0.0.1:8000/api/health  ·  docs: http://127.0.0.1:8000/docs
param([int]$Port = 8000)
. (Join-Path $PSScriptRoot "_common.ps1")
Assert-Python
& $Python (Join-Path $DemoDir "backend\main.py") --port $Port
