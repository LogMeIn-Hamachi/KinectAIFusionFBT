# Updating switches SteamVR's registration to this package; no user files are deleted.
function Invoke-KinectVrPathReg {
    param([string]$Registrar,[string]$Action,[string]$DriverPath)
    & $Registrar $Action $DriverPath
    if ($LASTEXITCODE -ne 0) { throw "SteamVR registration failed ($LASTEXITCODE): $Action" }
}
function Test-KinectSteamVrRunning {
    return [bool](Get-Process vrserver,vrmonitor,vrcompositor -ErrorAction SilentlyContinue)
}
function Update-KinectSteamVrDriver {
    param([string]$PackageRoot,[string]$PathsFile)
    $ErrorActionPreference='Stop'
    if (Test-KinectSteamVrRunning) { throw 'Close SteamVR completely, then run Update SteamVR Trackers.cmd again. No changes were made.' }
    $driverPath=[IO.Path]::GetFullPath((Join-Path $PackageRoot 'steamvr-driver\kinect_fbt')).TrimEnd('\','/')
    foreach ($relative in @('driver.vrdrivermanifest','bin\win64\driver_kinect_fbt.dll','resources\input\tracker_profile.json','resources\settings\default.vrsettings')) {
        $file=Join-Path $driverPath $relative
        if (-not (Test-Path -LiteralPath $file -PathType Leaf) -or (Get-Item -LiteralPath $file).Length -eq 0) {
            throw "Incomplete update package: missing or empty $relative. Extract the complete package first. No changes were made."
        }
    }
    $manifest=Get-Content -LiteralPath (Join-Path $driverPath 'driver.vrdrivermanifest') -Raw | ConvertFrom-Json
    if ($manifest.name -ne 'kinect_fbt') { throw 'Unexpected driver identity. No changes were made.' }
    if (-not (Test-Path -LiteralPath $PathsFile)) { throw 'SteamVR was not found. Install and start it once, then close it and retry.' }
    $paths=Get-Content -LiteralPath $PathsFile -Raw | ConvertFrom-Json
    $registrar=$null
    foreach ($runtime in $paths.runtime) {
        $candidate=Join-Path $runtime 'bin\win64\vrpathreg.exe'
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { $registrar=$candidate;break }
    }
    if (-not $registrar) { throw 'SteamVR registration utility was not found. No changes were made.' }
    $oldPaths=@();$alreadyHere=$false
    foreach ($registered in $paths.external_drivers) {
        if (-not $registered) { continue }
        $normalized=[IO.Path]::GetFullPath($registered).TrimEnd('\','/')
        if ($normalized -eq $driverPath) { $alreadyHere=$true;continue }
        $otherManifest=Join-Path $normalized 'driver.vrdrivermanifest'
        if (Test-Path -LiteralPath $otherManifest -PathType Leaf) {
            # Fail before mutation if a manifest cannot be inspected. Never guess
            # a driver's identity from a folder name or touch unrelated drivers.
            $other=Get-Content -LiteralPath $otherManifest -Raw | ConvertFrom-Json
            if ($other.name -eq 'kinect_fbt') { $oldPaths+=$normalized }
        }
    }
    $oldPaths=@($oldPaths | Select-Object -Unique)
    if ($alreadyHere -and $oldPaths.Count -eq 0) {
        Write-Host 'SteamVR already uses this folder. It will load the driver files currently here when started. No registration changes needed.'
        return
    }
    try {
        foreach ($old in $oldPaths) { Invoke-KinectVrPathReg $registrar 'removedriver' $old }
        if (-not $alreadyHere) { Invoke-KinectVrPathReg $registrar 'adddriver' $driverPath }
        $updated=Get-Content -LiteralPath $PathsFile -Raw | ConvertFrom-Json
        $registeredNow=@($updated.external_drivers | Where-Object { $_ } | ForEach-Object { [IO.Path]::GetFullPath($_).TrimEnd('\','/') })
        if ($driverPath -notin $registeredNow -or @($oldPaths | Where-Object { $_ -in $registeredNow }).Count) {
            throw 'SteamVR did not retain the requested registration.'
        }
    } catch {
        $failure=$_.Exception.Message;$rollbackErrors=@()
        if (-not $alreadyHere) {
            try { Invoke-KinectVrPathReg $registrar 'removedriver' $driverPath } catch { $rollbackErrors+=$_.Exception.Message }
        }
        foreach ($old in $oldPaths) {
            try { Invoke-KinectVrPathReg $registrar 'adddriver' $old } catch { $rollbackErrors+=$_.Exception.Message }
        }
        if ($rollbackErrors.Count) { throw "$failure Rollback also reported errors: $($rollbackErrors -join '; '). Use the installer in your old package to restore its registration." }
        throw "$failure Previous Kinect registrations restored. No driver files or settings were deleted."
    }
    Write-Host "SteamVR now uses: $driverPath"
    Write-Host 'Start SteamVR to load this driver. Keep this package in its current location. Old package files and personal settings were left intact.'
}
if ($MyInvocation.InvocationName -ne '.') {
    $ErrorActionPreference='Stop'
    Update-KinectSteamVrDriver -PackageRoot $PSScriptRoot -PathsFile (Join-Path $env:LOCALAPPDATA 'openvr\openvrpaths.vrpath')
}
