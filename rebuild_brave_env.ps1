param(
    [string]$PythonExe = "py",
    [string]$PythonVersion = "3.10",
    [string]$VenvName = ".venv"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$requirementsPath = Join-Path $repoRoot "requirements.txt"
$appPath = Join-Path $repoRoot "src\\main.py"
$venvPath = if ([System.IO.Path]::IsPathRooted($VenvName)) { $VenvName } else { Join-Path $repoRoot $VenvName }

Write-Host ""
Write-Host "BRAVE AI environment rebuild" -ForegroundColor Cyan
Write-Host "Repository: $repoRoot"
Write-Host "Virtual environment: $venvPath"

if ($PythonExe -eq "py") {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if (-not $launcher) {
        throw "Python launcher 'py' was not found. Install Python 3.10 or rerun this script with -PythonExe <full-path-to-python.exe>."
    }
    Write-Host "Creating a fresh virtual environment with py -$PythonVersion ..." -ForegroundColor Yellow
    & py "-$PythonVersion" -m venv $venvPath
}
else {
    $pythonCommand = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if (-not $pythonCommand -and -not (Test-Path $PythonExe)) {
        throw "Python executable '$PythonExe' was not found."
    }
    $resolvedPython = if ($pythonCommand) { $pythonCommand.Source } else { $PythonExe }
    Write-Host "Creating a fresh virtual environment with $resolvedPython ..." -ForegroundColor Yellow
    & $resolvedPython -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment creation failed. Expected interpreter at $venvPython"
}

Write-Host "Upgrading pip ..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip

Write-Host "Installing project dependencies ..." -ForegroundColor Yellow
& $venvPython -m pip install -r $requirementsPath

Write-Host ""
Write-Host "BRAVE AI environment is ready." -ForegroundColor Green
Write-Host "Activate it with:"
Write-Host "  $venvPath\\Scripts\\Activate.ps1"
Write-Host ""
Write-Host "Run the app with:"
Write-Host "  $venvPython -m streamlit run `"$appPath`""
