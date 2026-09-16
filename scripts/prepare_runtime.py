"""Assemble a fresh runnable development folder from pinned dependencies and LFS assets."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import os
from package_distribution import ROOT_DLLS, SAM_DLLS

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'release/dev')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError('Choose a new output folder; existing installations are never overwritten')
    torch = ROOT/'third_party/sam3d_env/Lib/site-packages/torch/lib'
    build = ROOT/'build/Release'
    vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))/'Microsoft Visual Studio/Installer/vswhere.exe'
    vs = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-property', 'installationPath'], text=True).strip()
    crts = sorted((Path(vs)/'VC/Redist/MSVC').glob('*/x64/Microsoft.VC*.CRT'), reverse=True)
    if not crts:
        raise RuntimeError('Visual Studio x64 redistributable runtime not found')
    sources = [ROOT/'third_party/trt', ROOT/'third_party/onnxruntime-win-x64-1.30.0/lib',
               ROOT/'third_party/openvr/bin/win64', crts[0]]
    files = {'KinectRGBD.exe': build/'KinectRGBD.exe'}
    for name in ROOT_DLLS:
        candidates = [p for folder in sources for p in folder.rglob(name)]
        if not candidates:
            raise RuntimeError('Missing runtime dependency: '+name)
        files[name] = candidates[0]
    for name in SAM_DLLS:
        files['sam3d-runtime/'+name] = build/name if name=='kf_sam3d_decoder.dll' else torch/name
    for model in ('sam3d', 'sam3d-optimized'):
        for p in (ROOT/'assets'/model).iterdir():
            if p.is_file(): files[f'assets/{model}/{p.name}'] = p
    for p in (ROOT/'driver').rglob('*'):
        if p.is_file(): files['steamvr-driver/'+p.relative_to(ROOT/'driver').as_posix()] = p
    files['steamvr-driver/kinect_fbt/bin/win64/driver_kinect_fbt.dll'] = build/'driver_kinect_fbt.dll'
    for p in (ROOT/'docs/licenses').iterdir():
        if p.is_file(): files['docs/licenses/'+p.name] = p
    for p in (ROOT/'docs/model-provenance').iterdir():
        if p.is_file(): files['docs/model-provenance/'+p.name] = p
    for p in (ROOT/'packaging/windows').iterdir():
        if p.is_file(): files[('docs/' if p.name in ('TROUBLESHOOTING.md','RELEASE-NOTES.md') else '')+p.name] = p
    missing = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in files.values() if not p.is_file()]
    if missing: raise RuntimeError('Missing build/dependency files: '+', '.join(missing))
    for rel, source in files.items():
        if source.suffix in ('.pt','.onnx','.data') and source.stat().st_size < 1024:
            raise RuntimeError('Model is an LFS pointer. Run git lfs pull: '+str(source))
        dest = out/rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source,dest)
    print(json.dumps({'output': str(out), 'files': len(files), 'camera_started': False}))

if __name__ == '__main__': main()
