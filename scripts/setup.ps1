$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3.12 -m venv .venv
    } else {
        python -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 or 3.13 and retry.' }
}
& '.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.venv\Scripts\python.exe' scripts/doctor.py
Write-Host 'Run: .\.venv\Scripts\python.exe -m streamlit run app.py'
