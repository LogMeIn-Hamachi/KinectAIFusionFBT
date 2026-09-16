# Development only. Does not install or alter USB/GPU drivers or system PATH.
param([switch]$CoreOnly)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$lock=Get-Content -LiteralPath (Join-Path $projectRoot 'dependencies.lock.json') -Raw | ConvertFrom-Json
foreach($asset in $lock.assets) {
    if ($CoreOnly -and -not $asset.core) { continue }
    $destination=Join-Path $projectRoot $asset.file
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
    if(!(Test-Path -LiteralPath $destination)) {
        Write-Host "Downloading pinned $($asset.name) from $($asset.url)"
        Invoke-WebRequest -Uri $asset.url -OutFile ($destination+'.part')
        if((Get-FileHash -LiteralPath ($destination+'.part') -Algorithm SHA256).Hash -ne $asset.sha256) {throw "Integrity mismatch: $($asset.name)"}
        Move-Item -LiteralPath ($destination+'.part') -Destination $destination
    }
    if((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $asset.sha256) {throw "Integrity mismatch: $($asset.name)"}
    if($asset.extract) {Expand-Archive -LiteralPath $destination -DestinationPath (Join-Path $projectRoot $asset.extract) -Force}
}
Write-Host 'Pinned dependencies verified. Sensor and GPU installations are unchanged.'
