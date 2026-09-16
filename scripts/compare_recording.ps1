param([Parameter(Mandatory=$true)][string]$Recording)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
$recordPath=(Resolve-Path -LiteralPath $Recording).Path
$out=Join-Path $projectRoot 'artifacts/ablation'
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach($mode in 'raw','filtered','fused') {
    & "$projectRoot/build/Release/kf_validate.exe" replay $recordPath "$projectRoot/assets/pose.onnx" "$out/$mode.csv" $mode
    if($LASTEXITCODE -ne 0){throw "Replay failed: $mode"}
}
Write-Host "Saved controlled ablation CSVs in $out. These are pose/uncertainty traces, not ground-truth accuracy measurements."
