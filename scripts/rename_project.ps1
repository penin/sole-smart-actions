<#
.SYNOPSIS
  Safely rename the local project folder BagOfWords -> SoleSmartActions (Windows).

.DESCRIPTION
  Run this AFTER closing VS Code, Claude Code, any terminal whose current folder is inside the project,
  and the demo server. Run it from a PowerShell window opened OUTSIDE the project folder, e.g.:

      cd "$env:USERPROFILE\OneDrive\Documents\ai-edge\Assignment"
      powershell -ExecutionPolicy Bypass -File .\BagOfWords\scripts\rename_project.ps1

  What it does (nothing destructive):
    1. checks the folders and that no running process is using the project (git, python, node)
    2. records `git status`, branches and HEAD before the rename
    3. renames the folder with Rename-Item (the .git directory, history, branches and uncommitted
       changes move with it unchanged)
    4. re-registers the editable install so `import sole_smart_actions` points at the new path
       (venv launchers such as pytest.exe keep the old path: use `python -m pytest`, or -RecreateVenv)
    5. verifies HEAD, branches and status are identical afterwards
  It never pushes, never touches the GitHub remote, and never deletes anything unless -RecreateVenv is given
  (which rebuilds .venv from requirements.lock).
#>
param(
    [string]$Parent = (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent),
    [string]$OldName = "BagOfWords",
    [string]$NewName = "SoleSmartActions",
    [switch]$RecreateVenv
)
$ErrorActionPreference = "Stop"
$old = Join-Path $Parent $OldName
$new = Join-Path $Parent $NewName

if (-not (Test-Path (Join-Path $old ".git"))) { throw "Not found: $old (with a .git folder)" }
if (Test-Path $new) { throw "$new already exists; refusing to overwrite it." }
if ((Get-Location).Path.StartsWith($old, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "This PowerShell window is inside the project. cd to $Parent first."
}

$busy = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^(python|pythonw|node|git|Code|claude)\.exe$' -and
    (($_.CommandLine -as [string]) -like "*$old*" -or ($_.ExecutablePath -as [string]) -like "$old*")
}
if ($busy) {
    $busy | Select-Object ProcessId, Name | Format-Table | Out-String | Write-Host
    throw "Close the processes above (VS Code, demo server, Python, Claude Code) and run again."
}

Push-Location $old
$before = @{ head = (git rev-parse HEAD); branches = (git branch --format "%(refname:short)") -join ","; status = (git status --porcelain) -join "`n" }
Pop-Location
Write-Host "HEAD $($before.head)  branches [$($before.branches)]  uncommitted entries: $(@($before.status -split "`n" | Where-Object { $_ }).Count)"

try {
    Rename-Item -LiteralPath $old -NewName $NewName
} catch {
    throw "Rename failed (a file is still open - often OneDrive sync, VS Code or a terminal): $($_.Exception.Message)"
}
Write-Host "Renamed to $new" -ForegroundColor Green

Push-Location $new
try {
    $after = @{ head = (git rev-parse HEAD); branches = (git branch --format "%(refname:short)") -join ","; status = (git status --porcelain) -join "`n" }
    foreach ($k in "head", "branches", "status") {
        if ($before[$k] -ne $after[$k]) { throw "Git $k changed during the rename - investigate before continuing." }
    }
    Write-Host "Git history, branches and working-tree status are unchanged." -ForegroundColor Green

    $py = Join-Path $new ".venv\Scripts\python.exe"
    if ($RecreateVenv -and (Test-Path ".venv")) {
        Write-Host "Rebuilding .venv from requirements.lock ..."
        Remove-Item -Recurse -Force ".venv"
    }
    if (-not (Test-Path $py)) {
        if (Get-Command py -ErrorAction SilentlyContinue) { & py -3.13 -m venv .venv } else { & python -m venv .venv }
        & $py -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu126
    }
    & $py -m pip install -e . --no-deps --quiet
    & $py -c "import sole_smart_actions, sole_intent; print('imports OK from', sole_smart_actions.__file__)"
    Write-Host "`nDone. Reopen VS Code on: $new" -ForegroundColor Green
    Write-Host "Optional (manual, on github.com): rename the repository; the remote URL keeps working via GitHub's redirect."
} finally {
    Pop-Location
}
