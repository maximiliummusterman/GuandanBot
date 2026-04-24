param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8765,
    [string]$Checkpoint
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$sitePackages = Join-Path $scriptDir ".venv311\Lib\site-packages"
$torchLib = Join-Path $sitePackages "torch\lib"

if (Test-Path $sitePackages) {
    $env:PYTHONPATH = $sitePackages
}

if (Test-Path $torchLib) {
    $env:PATH = "$torchLib;$env:PATH"
}

if ($Checkpoint) {
    $env:GUANDAN_CHECKPOINT = (Resolve-Path $Checkpoint).Path
}

$argsList = @("bot_api.py", "--host", $Host, "--port", $Port)

try {
    & py -3.12 @argsList
}
catch {
    & py @argsList
}
