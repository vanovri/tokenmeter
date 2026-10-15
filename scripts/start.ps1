# Start TokenMeter proxy (8787) and dashboard (8501) hidden. Safe to run repeatedly.
# Never prints NINEROUTER_API_KEY.
param([int]$ProxyPort = 8787, [int]$DashPort = 8501)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force (Join-Path $root 'logs') | Out-Null

if (-not $env:NINEROUTER_API_KEY) {
    $env:NINEROUTER_API_KEY = [Environment]::GetEnvironmentVariable('NINEROUTER_API_KEY', 'User')
}
if (-not $env:TOKENMETER_UPSTREAM) { $env:TOKENMETER_UPSTREAM = 'http://127.0.0.1:20128/v1' }
if (-not $env:TOKENMETER_REPLACE_AUTH) { $env:TOKENMETER_REPLACE_AUTH = 'tokenmeter-local' }
$env:TOKENMETER_DB = Join-Path $root 'tokenmeter.db'
$env:TOKENMETER_PRICES = Join-Path $root 'config\prices.yaml'

function Test-Listening([int]$port) {
    [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

if (-not (Test-Listening $ProxyPort)) {
    $pargs = @('-m', 'tokenmeter.serve', '--port', "$ProxyPort")
    # Also listen on the WSL vEthernet IP (NAT mode) so WSL apps can reach the proxy; never 0.0.0.0.
    $wsl = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.InterfaceAlias -like 'vEthernet (WSL*' } | Select-Object -First 1 -ExpandProperty IPAddress
    if ($wsl) { $pargs += @('--extra-host', $wsl) }
    Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList $pargs `
        -RedirectStandardOutput logs\proxy.out.log -RedirectStandardError logs\proxy.err.log
    Write-Output "proxy: http://127.0.0.1:$ProxyPort/v1$(if ($wsl) { " and http://${wsl}:$ProxyPort/v1 (WSL)" })"
} else { Write-Output "proxy already listening on $ProxyPort" }

if (-not (Test-Listening $DashPort)) {
    Start-Process -WindowStyle Hidden -FilePath $py `
        -ArgumentList @('-m', 'streamlit', 'run', 'dashboard.py', '--server.port', "$DashPort", '--server.headless', 'true', '--server.address', '127.0.0.1') `
        -RedirectStandardOutput logs\dash.out.log -RedirectStandardError logs\dash.err.log
    Write-Output "dashboard: http://127.0.0.1:$DashPort"
} else { Write-Output "dashboard already listening on $DashPort" }
