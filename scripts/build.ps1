param([switch]$Bootstrap, [switch]$CoreOnly)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    if ($Bootstrap) { & "$PSScriptRoot/bootstrap.ps1" -CoreOnly:$CoreOnly }
    & cmake --preset windows-x64
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
    if ($CoreOnly) {
        $targets=@('kf_steamvr_tests','kf_exposure_retry_tests','kf_tests','kf_sam3d_tests','kf_temporal_tests','kf_pose_continuity_tests','kf_body_tracker_tests','kf_controller_tests','kf_nlf_tests','kf_v2_tests')
        & cmake --build --preset release --target @targets
    } else { & cmake --build --preset release }
    if ($LASTEXITCODE -ne 0) { throw 'Native build failed' }
    & ctest --preset release
    if ($LASTEXITCODE -ne 0) { throw 'Native tests failed' }
    if (-not $CoreOnly) {
        & ./build/Release/kf_lifecycle.exe
        if ($LASTEXITCODE -ne 0) { throw 'Offline engine lifecycle checks failed' }
    }
    if (-not $CoreOnly) { Write-Host 'Build passed. Run scripts/prepare_runtime.py to assemble a runnable app.' }
} finally { Pop-Location }
