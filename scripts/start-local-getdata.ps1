[CmdletBinding()]
param(
    [ValidateRange(5, 120)]
    [int]$TimeoutSeconds = 60,

    [ValidateRange(1, 65535)]
    [int]$Port = 18082,

    [string]$DataDirectory
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$baseUrl = "http://127.0.0.1:$Port"

function Resolve-EffectiveDirectoryPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    $root = [IO.Path]::GetPathRoot($absolute)
    $effective = $root.TrimEnd([IO.Path]::DirectorySeparatorChar)
    foreach ($segment in $absolute.Substring($root.Length).Split([IO.Path]::DirectorySeparatorChar)) {
        if ([string]::IsNullOrEmpty($segment)) {
            continue
        }
        $effective = Join-Path $effective $segment
        if (Test-Path -LiteralPath $effective) {
            $item = Get-Item -LiteralPath $effective -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                $target = $item.ResolveLinkTarget($true)
                if ($null -eq $target) {
                    throw "Could not resolve reparse-point target: $effective"
                }
                $effective = $target.FullName
            }
        }
    }
    return [IO.Path]::GetFullPath($effective)
}

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Project Python executable is missing: $pythonExe"
}

if ([string]::IsNullOrWhiteSpace($DataDirectory)) {
    $timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $bytes = [byte[]]::new(4)
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    $suffix = [Convert]::ToHexString($bytes).ToLowerInvariant()
    $DataDirectory = Join-Path $projectRoot ".runtime\getdata\$timestamp-$suffix"
}
$resolvedDataDirectory = [IO.Path]::GetFullPath($DataDirectory)
$effectiveProjectRoot = Resolve-EffectiveDirectoryPath $projectRoot
$effectiveDataDirectory = Resolve-EffectiveDirectoryPath $resolvedDataDirectory
$relativeDataDirectory = [IO.Path]::GetRelativePath($effectiveProjectRoot, $effectiveDataDirectory)
if (
    $relativeDataDirectory -eq "." -or
    [IO.Path]::IsPathRooted($relativeDataDirectory) -or
    $relativeDataDirectory -eq ".." -or
    $relativeDataDirectory.StartsWith("..$([IO.Path]::DirectorySeparatorChar)")
) {
    throw "Getdata data directory must resolve beneath this project: $resolvedDataDirectory"
}
if (Test-Path -LiteralPath $resolvedDataDirectory) {
    throw "Getdata data directory already exists; a fresh directory is required: $resolvedDataDirectory"
}

$portProbe = [System.Net.Sockets.TcpListener]::new(
    [System.Net.IPAddress]::Loopback,
    $Port
)
try {
    $portProbe.Start()
} catch [System.Net.Sockets.SocketException] {
    throw "Getdata loopback port $Port is already occupied; refusing to start another process."
} finally {
    $portProbe.Stop()
}

$parentDirectory = [IO.Directory]::GetParent($resolvedDataDirectory).FullName
[IO.Directory]::CreateDirectory($parentDirectory) | Out-Null
$stdoutPath = "$resolvedDataDirectory.stdout.log"
$stderrPath = "$resolvedDataDirectory.stderr.log"
$commandLine = '""{0}" -m tn_governed.feed_server --port {1} --data-directory "{2}" 1> "{3}" 2> "{4}""' -f `
    $pythonExe, $Port, $resolvedDataDirectory, $stdoutPath, $stderrPath
$process = Start-Process `
    -FilePath $env:ComSpec `
    -ArgumentList @("/d", "/c", $commandLine) `
    -WorkingDirectory $projectRoot `
    -WindowStyle Hidden `
    -PassThru

try {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        if ($process.HasExited) {
            throw "Local getdata process $($process.Id) exited before readiness."
        }
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/health" -Method Get -TimeoutSec 3 | Out-Null
            $ready = $true
            break
        } catch {
            Start-Sleep -Milliseconds 250
        }
    }
    if (-not $ready) {
        throw "Local getdata did not become ready within $TimeoutSeconds seconds."
    }
} catch {
    $readinessFailure = $_
    $cleanupFailures = [System.Collections.Generic.List[string]]::new()
    if (-not $process.HasExited) {
        & taskkill /PID $process.Id /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) {
            $cleanupFailures.Add("owned process tree $($process.Id) could not be stopped")
        }
    }
    if (-not $process.WaitForExit(5000)) {
        $cleanupFailures.Add("owned process $($process.Id) did not exit")
    }
    $releaseProbe = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $Port
    )
    try {
        $releaseProbe.Start()
    } catch [System.Net.Sockets.SocketException] {
        $cleanupFailures.Add("loopback port $Port remains bound")
    } finally {
        $releaseProbe.Stop()
    }
    if ($cleanupFailures.Count -gt 0) {
        throw "$readinessFailure Cleanup verification failed: $($cleanupFailures -join '; ')."
    }
    throw $readinessFailure
}

Write-Output "Local getdata is ready. PID: $($process.Id); base URL: $baseUrl; data directory: $resolvedDataDirectory"
