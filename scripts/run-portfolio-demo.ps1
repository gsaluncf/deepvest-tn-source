[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$RunDirectory,

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$UnityUrl = "http://127.0.0.1:18080",

    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$FeedUrl = "http://127.0.0.1:18082",

    [Parameter(Mandatory)]
    [ValidateRange(1, 65535)]
    [int]$LlamaPort
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:TN_LLAMA_API_KEY)) {
    throw "TN_LLAMA_API_KEY must be set to a nonempty local API key."
}

$containmentNames = @(
    "PYTHONDONTWRITEBYTECODE",
    "TN_NO_LINK",
    "TN_NO_STDOUT",
    "TN_AUTOINIT_QUIET"
)
$previous = @{}
foreach ($name in $containmentNames) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$previousPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "Process")
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$siblingPaths = @(
    (Resolve-Path -LiteralPath (Join-Path $projectRoot "src")).Path,
    (Resolve-Path -LiteralPath (Join-Path $projectRoot "dependencies\tn-polars")).Path,
    (Resolve-Path -LiteralPath (Join-Path $projectRoot "dependencies\tn-llama-client")).Path
)

try {
    foreach ($name in $containmentNames) {
        [Environment]::SetEnvironmentVariable($name, "1", "Process")
    }

    $pythonPaths = @($siblingPaths)
    if (-not [string]::IsNullOrWhiteSpace($previousPythonPath)) {
        $pythonPaths += $previousPythonPath
    }
    [Environment]::SetEnvironmentVariable(
        "PYTHONPATH",
        ($pythonPaths -join [IO.Path]::PathSeparator),
        "Process"
    )

    $commandPath = Join-Path $projectRoot ".venv\Scripts\tn-governed.exe"
    if (-not (Test-Path -LiteralPath $commandPath -PathType Leaf)) {
        throw "tn-governed command is missing; recreate the project environment."
    }

    & $commandPath @(
        "portfolio-demo",
        "--run-directory", $RunDirectory,
        "--unity-url", $UnityUrl,
        "--feed-url", $FeedUrl,
        "--llama-port", [string]$LlamaPort
    )
    if ($LASTEXITCODE -ne 0) {
        throw "tn-governed portfolio-demo exited with code $LASTEXITCODE."
    }
} finally {
    foreach ($name in $containmentNames) {
        if ($null -eq $previous[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($name, $previous[$name], "Process")
        }
    }
    if ($null -eq $previousPythonPath) {
        Remove-Item -LiteralPath "Env:PYTHONPATH" -ErrorAction SilentlyContinue
    } else {
        [Environment]::SetEnvironmentVariable("PYTHONPATH", $previousPythonPath, "Process")
    }
}
