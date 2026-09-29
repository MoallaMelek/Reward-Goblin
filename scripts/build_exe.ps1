# Build RewardGoblin.exe (one-folder PyInstaller app with the full backend, PyTorch CPU included).
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1 [-Python C:\path\to\python.exe] [-Out C:\RewardGoblin]
#
# Use an isolated venv (no global site-packages) with: torch (CPU) gymnasium stable-baselines3 pymunk fastapi uvicorn pyinstaller.
# Keep -Out on a SHORT path: Pymunk's DLL fails to load from very long Windows paths.
param(
  [string]$Python = "$env:USERPROFILE\.venvs\rgb\Scripts\python.exe",
  [string]$Out = "$env:USERPROFILE\RewardGoblin"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Parent = Split-Path -Parent $Out
$Name = Split-Path -Leaf $Out
$Work = Join-Path $env:TEMP "rgob-pyinstaller"

& $Python -m PyInstaller (Join-Path $Repo "launcher.py") `
  --name $Name --onedir --console --noconfirm --clean `
  --icon (Join-Path $Repo "docs\goblin.ico") `
  --paths $Repo --distpath $Parent --workpath $Work --specpath $Work `
  --collect-all pymunk --collect-data stable_baselines3 `
  --collect-submodules uvicorn --collect-submodules backend `
  --exclude-module matplotlib --exclude-module tkinter --exclude-module pytest --exclude-module IPython
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# Writable data next to the exe: gallery results, replays, experiment configs, the UI, and policy weights.
$Data = Join-Path $Out "data"
New-Item -ItemType Directory -Force $Data | Out-Null
foreach ($d in "frontend", "experiments", "configs", "runs", "recordings") {
  Copy-Item -Recurse -Force (Join-Path $Repo $d) (Join-Path $Data $d)
}
New-Item -ItemType Directory -Force (Join-Path $Data "models") | Out-Null
Get-ChildItem (Join-Path $Repo "models") -Filter *.zip -ErrorAction SilentlyContinue | Copy-Item -Destination (Join-Path $Data "models")
Remove-Item -Recurse -Force (Join-Path $Data "runs\_jobs") -ErrorAction SilentlyContinue
Write-Host "Built $Out\$Name.exe"
