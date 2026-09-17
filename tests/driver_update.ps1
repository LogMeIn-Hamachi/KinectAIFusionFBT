$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $projectRoot 'packaging/windows/Update-SteamVR-Trackers.ps1')
$testRoot=Join-Path $projectRoot ('artifacts/driver-update-tests-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null
$package=Join-Path $testRoot 'new package'
$target=Join-Path $package 'steamvr-driver/kinect_fbt'
$runtime=Join-Path $testRoot 'runtime'
$script:pathsFile=Join-Path $testRoot 'openvrpaths.vrpath'
foreach ($sub in @('bin/win64','resources/input','resources/settings')) { New-Item -ItemType Directory -Path (Join-Path $target $sub) -Force | Out-Null }
'{"name":"kinect_fbt"}' | Set-Content (Join-Path $target 'driver.vrdrivermanifest')
foreach ($file in @('bin/win64/driver_kinect_fbt.dll','resources/input/tracker_profile.json','resources/settings/default.vrsettings')) { 'fixture-only' | Set-Content (Join-Path $target $file) }
New-Item -ItemType Directory -Path (Join-Path $runtime 'bin/win64') -Force | Out-Null
'never executed' | Set-Content (Join-Path $runtime 'bin/win64/vrpathreg.exe')
$old=Join-Path $testRoot 'old package'
$other=Join-Path $testRoot 'unrelated driver'
foreach ($folder in @($old,$other)) { New-Item -ItemType Directory -Path $folder | Out-Null }
'{"name":"kinect_fbt"}' | Set-Content (Join-Path $old 'driver.vrdrivermanifest')
'{"name":"unrelated"}' | Set-Content (Join-Path $other 'driver.vrdrivermanifest')
function Save-Fixture { @{runtime=@($runtime);external_drivers=@($script:registered)} | ConvertTo-Json | Set-Content $script:pathsFile }
function Reset-Fixture($entries) { $script:registered=@($entries);$script:calls=0;$script:running=$false;$script:failOnce=$false;Save-Fixture }
function Test-KinectSteamVrRunning { return $script:running }
function Invoke-KinectVrPathReg($Registrar,$Action,$DriverPath) {
    $script:calls++
    if ($Action -eq 'adddriver') {
        if ($script:failOnce -and $DriverPath -eq $target) { $script:failOnce=$false;throw 'Simulated registration failure' }
        $script:registered=@($script:registered+$DriverPath | Select-Object -Unique)
    } else { $script:registered=@($script:registered | Where-Object { $_ -ne $DriverPath }) }
    Save-Fixture
}
function Check($ok,$message) { if (-not $ok) { throw $message } }
function Update-Fixture { Update-KinectSteamVrDriver -PackageRoot $package -PathsFile $script:pathsFile }
Reset-Fixture @($old,$other)
Update-Fixture
Check ($target -in $script:registered -and $old -notin $script:registered -and $other -in $script:registered) 'Wrong registrations after update'
Check (Test-Path (Join-Path $old 'driver.vrdrivermanifest')) 'Old files were removed'
Reset-Fixture @($target,$other)
Update-Fixture
Check ($script:calls -eq 0) 'Same-folder update was not idempotent'
Reset-Fixture @($other)
Update-Fixture
Check ($target -in $script:registered -and $other -in $script:registered) 'First install via updater failed'
Reset-Fixture @($target,$old,$other)
Update-Fixture
Check ($script:calls -eq 1 -and $old -notin $script:registered -and $target -in $script:registered) 'Duplicate Kinect cleanup failed'
Reset-Fixture @($old,$other)
$script:running=$true
try { Update-Fixture;throw 'Expected running-process refusal' } catch { Check ($_.Exception.Message -like 'Close SteamVR*') 'Unexpected running-process error' }
Check ($script:calls -eq 0) 'Running SteamVR was modified'
Reset-Fixture @($old,$other)
try { Update-KinectSteamVrDriver -PackageRoot (Join-Path $testRoot 'incomplete') -PathsFile $script:pathsFile;throw 'Expected incomplete-package refusal' } catch { Check ($_.Exception.Message -like 'Incomplete update package*') 'Unexpected package validation error' }
Check ($script:calls -eq 0) 'Incomplete package changed registration'
Reset-Fixture @($old,$other)
$script:failOnce=$true
try { Update-Fixture;throw 'Expected registration failure' } catch { Check ($_.Exception.Message -like '*Previous Kinect registrations restored*') 'Rollback did not report restoration' }
Check ($old -in $script:registered -and $other -in $script:registered -and $target -notin $script:registered) 'Rollback did not restore old registration'
Write-Host 'PASS: new-folder update, same-folder retry, first install, duplicate registration, running SteamVR, incomplete package, rollback and unrelated-driver preservation. Only temporary fixtures were used.'
