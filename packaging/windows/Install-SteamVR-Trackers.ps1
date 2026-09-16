param([switch]$Remove)
$ErrorActionPreference = 'Stop'
$driverPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'steamvr-driver\kinect_fbt'))
if (-not (Test-Path -LiteralPath (Join-Path $driverPath 'driver.vrdrivermanifest'))) {
    throw 'Extract the complete ZIP first. Keep this installer beside KinectRGBD.exe and the steamvr-driver folder.'
}
$pathsFile = Join-Path $env:LOCALAPPDATA 'openvr\openvrpaths.vrpath'
if (-not (Test-Path -LiteralPath $pathsFile)) {
    throw 'SteamVR was not found. Install and start SteamVR once, then run this installer again.'
}
$paths = Get-Content -LiteralPath $pathsFile -Raw | ConvertFrom-Json
$registrar = $null
foreach ($runtimePath in $paths.runtime) {
    $candidate = Join-Path $runtimePath 'bin\win64\vrpathreg.exe'
    if (Test-Path -LiteralPath $candidate) { $registrar = $candidate; break }
}
if (-not $registrar) { throw 'SteamVR registration utility was not found. Start SteamVR once and try again.' }
if (-not $Remove) {
    foreach ($registered in $paths.external_drivers) {
        if ([IO.Path]::GetFullPath($registered).TrimEnd('\','/') -eq $driverPath.TrimEnd('\','/')) { continue }
        $manifest = Join-Path $registered 'driver.vrdrivermanifest'
        if (Test-Path -LiteralPath $manifest) {
            $driver = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
            if ($driver.name -eq 'kinect_fbt') {
                throw "Another Kinect tracker folder is registered: $registered. Run Remove SteamVR Trackers.cmd there (or its Install-SteamVR-Trackers.ps1 with -Remove), then install this copy."
            }
        }
    }
}
$action = if ($Remove) { 'removedriver' } else { 'adddriver' }
& $registrar $action $driverPath
if ($LASTEXITCODE -ne 0) { throw "SteamVR registration failed ($LASTEXITCODE)." }
Write-Host 'Registration updated. Restart SteamVR to apply the change.'
