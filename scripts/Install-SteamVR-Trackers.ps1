param([switch]$Remove)
$ErrorActionPreference = 'Stop'
$driverPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'steamvr-driver\kinect_fbt'))
if (-not (Test-Path -LiteralPath (Join-Path $driverPath 'driver.vrdrivermanifest'))) {
    throw 'Keep this installer beside KinectRGBD.exe and the steamvr-driver folder.'
}
$pathsFile = Join-Path $env:LOCALAPPDATA 'openvr\openvrpaths.vrpath'
$paths = Get-Content -LiteralPath $pathsFile -Raw | ConvertFrom-Json
$registrar = $null
foreach ($runtimePath in $paths.runtime) {
    $candidate = Join-Path $runtimePath 'bin\win64\vrpathreg.exe'
    if (Test-Path -LiteralPath $candidate) { $registrar = $candidate; break }
}
if (-not $registrar) { throw 'SteamVR was not found. Install and start SteamVR once, then run this installer again.' }
$action = if ($Remove) { 'removedriver' } else { 'adddriver' }
& $registrar $action $driverPath
if ($LASTEXITCODE -ne 0) { throw "SteamVR registration failed ($LASTEXITCODE)." }
Write-Host 'Registration updated. Restart SteamVR to apply the change.'
