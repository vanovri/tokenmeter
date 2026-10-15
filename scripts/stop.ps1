# Stop the TokenMeter processes listening on the proxy/dashboard ports.
# (A venv python.exe on Windows is a launcher that runs the base interpreter as a child,
#  so we match on the command line and also stop the venv launcher parent.)
param([int]$ProxyPort = 8787, [int]$DashPort = 8501)
$root = Split-Path -Parent $PSScriptRoot
foreach ($port in @($ProxyPort, $DashPort)) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
            $p = Get-CimInstance Win32_Process -Filter "ProcessId=$_"
            if (-not $p) { return }
            if ($p.CommandLine -match 'tokenmeter|dashboard\.py') {
                $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($p.ParentProcessId)"
                Stop-Process -Id $p.ProcessId -Force
                if ($parent -and $parent.ExecutablePath -like "$root*") { Stop-Process -Id $parent.ProcessId -Force -ErrorAction SilentlyContinue }
                Write-Output "stopped pid $($p.ProcessId) on port $port"
            } else { Write-Output "port $port is used by another program ($($p.Name)) - left alone" }
        }
}
