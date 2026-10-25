[CmdletBinding()]
param(
    [string]$RunDirectory,
    [string]$UnityUrl = "http://127.0.0.1:18080",
    [ValidateRange(1, 65535)]
    [int]$LlamaPort = 18182
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$commandPath = Join-Path $projectRoot ".venv\Scripts\deepvest-tn.exe"
if (-not (Test-Path -LiteralPath $commandPath -PathType Leaf)) {
    throw "deepvest-tn command is missing; create the project environment first."
}
if ([string]::IsNullOrWhiteSpace($env:TN_LLAMA_API_KEY)) {
    throw "TN_LLAMA_API_KEY must be set to a nonempty local API key."
}
if ([string]::IsNullOrWhiteSpace($RunDirectory)) {
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $RunDirectory = Join-Path $projectRoot "experiments\results\reference-$stamp"
}

$previousPythonPath = $env:PYTHONPATH
$paths = @(
    (Join-Path $projectRoot "src"),
    (Join-Path $projectRoot "dependencies\tn-polars"),
    (Join-Path $projectRoot "dependencies\tn-llama-client")
)
if (-not [string]::IsNullOrWhiteSpace($previousPythonPath)) {
    $paths += $previousPythonPath
}
try {
    $env:PYTHONPATH = $paths -join [IO.Path]::PathSeparator
    & $commandPath @(
        "reference-path",
        "--run-directory", ([IO.Path]::GetFullPath($RunDirectory)),
        "--unity-url", $UnityUrl,
        "--llama-port", [string]$LlamaPort
    )
    if ($LASTEXITCODE -ne 0) {
        throw "DeepVest-TN reference path exited with code $LASTEXITCODE."
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
