[CmdletBinding()]
param([ValidateRange(1, 65535)][int]$LlamaPort = 18182)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

function Stop-Listener([int]$Port) {
    $owners = @(
        & "$env:SystemRoot\System32\netstat.exe" -ano -p TCP |
            ForEach-Object {
                if ($_ -match "^\s*TCP\s+\S*[\.:]$Port\s+\S+\s+LISTENING\s+(\d+)\s*$") {
                    [int]$Matches[1]
                }
            } |
            Sort-Object -Unique
    )
    foreach ($ownerPid in $owners) {
        Stop-Process -Id $ownerPid -Force -ErrorAction Stop
        Write-Host "Stopped PID $ownerPid on port $Port."
    }
}

function Start-Unity {
    & (Join-Path $PSScriptRoot "start-local-unity.ps1") -Port 18080
}

function Start-Reference {
    if ([string]::IsNullOrWhiteSpace($env:TN_LLAMA_API_KEY)) {
        $env:TN_LLAMA_API_KEY = "deepvest-tn-local"
    }
    & (Join-Path $PSScriptRoot "run-reference-path.ps1") -LlamaPort $LlamaPort
}

function Stop-All {
    Stop-Listener 18080
    Stop-Listener 18081
    Stop-Listener $LlamaPort
}

while ($true) {
    Write-Host ""
    Write-Host "DeepVest-TN local reference path"
    Write-Host "  1  Start Unity"
    Write-Host "  2  Run full reference path (TN GetData -> Polars -> DeepVest -> tn-llama)"
    Write-Host "  3  Start all (Unity, then full reference path)"
    Write-Host "  4  Restart Unity"
    Write-Host "  5  Restart full reference path"
    Write-Host "  6  Kill Unity"
    Write-Host "  7  Kill tn-llama"
    Write-Host "  8  Kill all"
    Write-Host "  0  Exit"
    $choice = Read-Host "Choose"
    switch ($choice) {
        "1" { Start-Unity }
        "2" { Start-Reference }
        "3" { Start-Unity; Start-Reference }
        "4" { Stop-Listener 18080; Stop-Listener 18081; Start-Unity }
        "5" { Stop-Listener $LlamaPort; Start-Reference }
        "6" { Stop-Listener 18080; Stop-Listener 18081 }
        "7" { Stop-Listener $LlamaPort }
        "8" { Stop-All }
        "0" { return }
        default { Write-Host "Choose 0 through 8." }
    }
}
