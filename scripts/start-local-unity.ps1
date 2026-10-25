param(
    [ValidateRange(5, 120)]
    [int]$TimeoutSeconds = 60,

    [ValidateRange(1, 65534)]
    [int]$Port = 8080,

    [string]$UnityRoot = $env:UNITY_HOME,
    [string]$JavaExe = "java"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($UnityRoot)) { throw "Set UNITY_HOME or pass -UnityRoot." }
$javaExe = (Get-Command $JavaExe -ErrorAction Stop).Source
$serverArgs = Join-Path $UnityRoot "server/target/server.args"
$internalPort = $Port + 1
$baseUrl = "http://127.0.0.1:$Port"
$readinessUrl = "$baseUrl/api/2.1/unity-catalog/catalogs?max_results=1"

foreach ($requiredPath in @($javaExe, $serverArgs)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required local Unity path is missing: $requiredPath"
    }
}

foreach ($portBinding in @(
    [pscustomobject]@{ Name = "public"; Number = $Port },
    [pscustomobject]@{ Name = "internal"; Number = $internalPort }
)) {
    $portProbe = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $portBinding.Number
    )
    try {
        $portProbe.Start()
    } catch [System.Net.Sockets.SocketException] {
        throw "$($portBinding.Name) loopback port $($portBinding.Number) is already occupied; refusing to start another Unity process."
    } finally {
        $portProbe.Stop()
    }
}

$process = Start-Process `
    -FilePath $javaExe `
    -ArgumentList "@$serverArgs", "--port", "$Port" `
    -WorkingDirectory $unityRoot `
    -WindowStyle Hidden `
    -PassThru

$ready = $false
try {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($process.HasExited) {
            throw "Local Unity process $($process.Id) exited before readiness."
        }
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $readinessUrl -Method Get -TimeoutSec 3 | Out-Null
            $ready = $true
            break
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $ready) {
        throw "Local Unity did not become ready within $TimeoutSeconds seconds."
    }
} catch {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -ErrorAction SilentlyContinue
    }
    throw
}

Write-Output "Local Unity is ready. PID: $($process.Id); base URL: $baseUrl"
