$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path -LiteralPath $venvActivate)) {
    throw "Python environment is missing. Recreate $projectRoot\.venv with Python 3.12."
}

. $venvActivate

$paths = @(
    (Join-Path $projectRoot "src"),
    (Join-Path $projectRoot "dependencies\tn-polars"),
    (Join-Path $projectRoot "dependencies\tn-llama-client")
)
$env:PYTHONPATH = ($paths + @($env:PYTHONPATH | Where-Object { $_ })).Where({ $_ }) -join [IO.Path]::PathSeparator

python -c "import polars as pl; print('DeepVest-TN experiment environment active; Polars', pl.__version__)"
