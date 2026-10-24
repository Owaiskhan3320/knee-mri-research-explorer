param([string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    $taskCandidates = @()
    if ($PythonPath) {
        $taskCandidates += $PythonPath
    } else {
        $taskPathPython = Get-Command python.exe -ErrorAction SilentlyContinue
        if ($taskPathPython) { $taskCandidates += $taskPathPython.Source }
        if ($env:LOCALAPPDATA) {
            $taskCandidates += Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
        }
        $taskLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
        if ($taskLauncher) {
            $taskRegisteredPython = & $taskLauncher.Source -3.12 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0) { $taskCandidates += $taskRegisteredPython }
        }
    }
    $taskBasePython = $null
    foreach ($taskCandidate in $taskCandidates) {
        if (-not (Test-Path -LiteralPath $taskCandidate -PathType Leaf)) { continue }
        $taskVersion = & $taskCandidate -c 'import sys; print(str(sys.version_info.major)+"."+str(sys.version_info.minor))' 2>$null
        if ($LASTEXITCODE -eq 0 -and $taskVersion -eq '3.12') { $taskBasePython = $taskCandidate; break }
    }
    if (-not $taskBasePython) { throw 'Install Python 3.12, or run setup.ps1 -PythonPath C:\path\to\python.exe.' }
    & $taskBasePython -m venv '.venv'
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Python 3.12 environment.' }
}
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$taskIs312 = & $taskPython -c 'import sys; print(sys.version_info[:2] == (3, 12))'
if ($LASTEXITCODE -ne 0 -or $taskIs312 -ne 'True') { throw 'This project requires a Python 3.12 environment.' }
& $taskPython -m pip install --disable-pip-version-check --no-input --timeout 30 --retries 1 'torch==2.11.0' 'torchvision==0.26.0' --index-url 'https://download.pytorch.org/whl/cpu'
if ($LASTEXITCODE -ne 0) { throw 'CPU PyTorch installation failed.' }
$taskRequirements = if (Test-Path -LiteralPath 'requirements-lock.txt') { 'requirements-lock.txt' } else { 'requirements.txt' }
& $taskPython -m pip install --disable-pip-version-check --no-input --timeout 30 --retries 1 -r $taskRequirements
if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed.' }
Write-Host 'Local setup complete. Run .\start.ps1. No paid service is configured.'
if (-not (Test-Path -LiteralPath 'models\deploy.pt')) {
    Write-Host 'Private weights are not distributed. See models\README.md before running inference.'
}
