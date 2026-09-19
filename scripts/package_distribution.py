"""Build a clean, explicit-allowlist Windows distribution. Never copy a live folder recursively."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'release/dev'
TEMPLATES = ROOT / 'packaging/windows'
VERSION = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()
if not re.fullmatch(r'\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?', VERSION):
    raise RuntimeError('Invalid VERSION file')
NAME = f'KinectAIFusionFBT-v{VERSION}-Windows-x64'
STAGE = ROOT / 'release' / NAME
ARCHIVE = ROOT / 'release' / (NAME + '.zip')
REPORT = ROOT / 'artifacts/packaging' / NAME

ROOT_DLLS = '''concrt140.dll msvcp140_1.dll msvcp140_2.dll msvcp140_atomic_wait.dll
msvcp140_codecvt_ids.dll msvcp140.dll onnxruntime_providers_nv_tensorrt_rtx.dll
onnxruntime_providers_shared.dll onnxruntime.dll openvr_api.dll tensorrt_onnxparser_rtx_1_6.dll
tensorrt_plugins.dll tensorrt_rtx_1_6.dll vccorlib140.dll vcruntime140_1.dll
vcruntime140_threads.dll vcruntime140.dll'''.split()
SAM_DLLS = '''c10_cuda.dll c10.dll caffe2_nvrtc.dll cublas64_12.dll cublasLt64_12.dll
cudart64_12.dll cudnn_adv64_9.dll cudnn_cnn64_9.dll cudnn_engines_precompiled64_9.dll
cudnn_engines_runtime_compiled64_9.dll cudnn_graph64_9.dll cudnn_heuristic64_9.dll
cudnn_ops64_9.dll cudnn64_9.dll cufft64_11.dll cufftw64_11.dll cupti64_2025.1.1.dll
curand64_10.dll cusolver64_11.dll cusolverMg64_11.dll cusparse64_12.dll kf_sam3d_decoder.dll
libiomp5md.dll libiompstubs5md.dll nvJitLink_120_0.dll nvperf_host.dll nvrtc-builtins64_128.dll
nvrtc64_120_0.dll nvToolsExt64_1.dll shm.dll torch_cpu.dll torch_cuda.dll torch_global_deps.dll
torch.dll uv.dll zlibwapi.dll'''.split()
LICENSES = '''CUDA-12.8-EULA.html cuDNN-License.html DINOv3-LICENSE.txt Fast-SAM-MIT.txt
MHR-LICENSE.txt Momentum-LICENSE.txt NUISENSOR-LICENSE.txt NVIDIA-EP-LICENSE.txt
NVIDIA-EP-ThirdPartyNotices.txt ONNXRuntime-LICENSE.txt ONNXRuntime-ThirdPartyNotices.txt
OpenVR-LICENSE.txt PyTorch-LICENSE.txt PyTorch-NOTICE.txt SAM-checkpoint-LICENSE.txt
SAM-LICENSE.txt TensorRT-RTX-Acknowledgements.txt TensorRT-RTX-SLA.html'''.split()

def selection():
    files = {'KinectRGBD.exe': ROOT / 'build/Release/KinectRGBD.exe'}
    for name in ROOT_DLLS:
        files[name] = SOURCE / name
    for name in SAM_DLLS:
        files['sam3d-runtime/' + name] = (ROOT / 'build/Release' / name if name == 'kf_sam3d_decoder.dll'
                                        else SOURCE / 'sam3d-runtime' / name)
    for model in ('sam3d', 'sam3d-optimized'):
        for name in ('backbone.onnx', 'backbone.onnx.data', 'backbone.onnx.files.sha256', 'decoder.pt'):
            rel = f'assets/{model}/{name}'
            files[rel] = SOURCE / rel
    files['assets/sam3d-optimized/conversion.json'] = SOURCE / 'assets/sam3d-optimized/conversion.json'
    for name in LICENSES:
        rel = 'docs/licenses/' + name
        files[rel] = SOURCE / rel
    for name in ('README.md', 'START-HERE.html', 'THIRD_PARTY_NOTICES.md',
                 'Install-SteamVR-Trackers.ps1', 'Install SteamVR Trackers.cmd', 'Remove SteamVR Trackers.cmd',
                 'Update-SteamVR-Trackers.ps1', 'Update SteamVR Trackers.cmd'):
        files[name] = TEMPLATES / name
    for name in ('TROUBLESHOOTING.md', 'RELEASE-NOTES.md'):
        files['docs/' + name] = TEMPLATES / name
    for rel in ('driver.vrdrivermanifest',
                'resources/input/tracker_profile.json', 'resources/settings/default.vrsettings'):
        key = 'steamvr-driver/kinect_fbt/' + rel
        files[key] = ROOT / 'driver/kinect_fbt' / rel
    files['steamvr-driver/kinect_fbt/bin/win64/driver_kinect_fbt.dll'] = ROOT / 'build/Release/driver_kinect_fbt.dll'
    files['docs/model-provenance/checkpoint.json'] = ROOT / 'docs/model-provenance/checkpoint.json'
    files['docs/model-provenance/encoder-fp8.json'] = ROOT / 'docs/model-provenance/encoder-fp8.json'
    return files

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    global SOURCE, NAME, STAGE, ARCHIVE, REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('stage', 'archive', 'refresh'))
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--name', default=NAME)
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in ('.','..'):
        raise RuntimeError('Package name must be one directory name')
    SOURCE=args.source.resolve(); NAME=args.name; STAGE=ROOT/'release'/NAME; ARCHIVE=ROOT/'release'/(NAME+'.zip')
    REPORT=ROOT/'artifacts/packaging'/NAME; REPORT.mkdir(parents=True,exist_ok=True)
    files = selection()
    if args.action == 'stage':
        if STAGE.exists():
            raise RuntimeError('Refusing to overwrite an existing release folder')
        missing = [str(p) for p in files.values() if not p.is_file()]
        if missing:
            raise RuntimeError('Missing required files: ' + repr(missing))
        for rel, source in files.items():
            target = STAGE / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        prerequisites = {
            'schema': 2, 'release': 'v' + VERSION, 'platform': 'Windows x64; tested Windows 11',
            'gpu': 'NVIDIA required; tested RTX 5070 Ti; CUDA 12.8 decoder and TensorRT RTX 1.6 encoder',
            'sensor': ['Kinect v2 + powered USB 3 adapter + installed Microsoft Kinect20 runtime',
                       'Kinect v1 + powered adapter + installed Microsoft Kinect SDK 1.8'],
            'vr': 'SteamVR, headset and two positional controllers; waist and feet by default; optional knees, elbows and chest',
            'models': ['SAM optimized (default)', 'SAM original'],
            'external_installations': ['Microsoft Kinect runtime/SDK', 'NVIDIA display driver', 'SteamVR'],
            'not_required': ['Python', 'CUDA developer toolkit', 'Visual Studio'],
            'personal_data_included': False,
        }
        (STAGE / 'prerequisites.json').write_text(json.dumps(prerequisites, indent=2) + '\n', encoding='utf-8')
        print(f'Staged {len(files)+1} files in {STAGE}', flush=True)
        return

    if ARCHIVE.exists() and args.action != 'refresh':
        raise RuntimeError('Refusing to replace an existing archive')
    if args.action == 'refresh' and not ARCHIVE.exists():
        raise RuntimeError('No archive to refresh')
    expected = set(files) | {'prerequisites.json'}
    actual = {p.relative_to(STAGE).as_posix() for p in STAGE.rglob('*') if p.is_file()}
    if args.action == 'refresh':
        actual.discard('package-sha256.json')
    if actual != expected:
        raise RuntimeError(f'Allowlist mismatch: unexpected={actual-expected}; missing={expected-actual}')
    # Verify model assets against the manifests consumed by the native loader.
    for model in ('sam3d', 'sam3d-optimized'):
        folder = STAGE / 'assets' / model
        for line in (folder / 'backbone.onnx.files.sha256').read_text(encoding='utf-8').splitlines():
            sha, name = line.split('  ', 1)
            if Path(name).name != name or digest(folder / name) != sha:
                raise RuntimeError('Model integrity failure: ' + model + '/' + name)
    records = []
    for rel in sorted(expected):
        path = STAGE / rel
        records.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': digest(path)})
        if path.suffix in {'.md', '.json', '.html', '.txt', '.ps1', '.cmd'} and 'docs/licenses/' not in rel:
            text = path.read_text(encoding='utf-8')
            if any(s in text.lower() for s in ('c:/users/', 'c:\\users/', 'c:\\users\\')):
                raise RuntimeError('Personal data found in ' + rel)
    manifest = {'release': 'v' + VERSION, 'scope': 'All distributed files except this manifest; locally generated files are not included', 'files': records}
    (STAGE / 'package-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Integrity/allowlist checks passed; creating full ZIP', flush=True)
    if args.action == 'refresh':
        # Copy unchanged compressed entries while replacing the small updated files.
        # An explicit full content/hash check below detects skipped updates as well.
        seven_zip = Path('C:/Program Files/7-Zip/7z.exe')
        subprocess.run([str(seven_zip), 'u', str(ARCHIVE), NAME, '-tzip', '-mx=1', '-y'],
                       cwd=STAGE.parent, check=True)
    else:
        with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            for i, rel in enumerate(sorted(expected | {'package-sha256.json'})):
                z.write(STAGE / rel, NAME + '/' + rel)
                if i % 10 == 0:
                    print(f'Archived {i+1}/{len(expected)+1} files', flush=True)
    print('Verifying every ZIP entry against its SHA-256 and CRC', flush=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        if len(members) != len(expected)+1 or {i.filename for i in members} != {NAME+'/'+p for p in expected | {'package-sha256.json'}}:
            raise RuntimeError('Archive integrity verification failed')
        hashes = {r['path']: r['sha256'] for r in records}
        hashes['package-sha256.json'] = digest(STAGE/'package-sha256.json')
        for rel, sha in hashes.items():
            h = hashlib.sha256()
            with z.open(NAME+'/'+rel) as f:
                for block in iter(lambda: f.read(8 << 20), b''):
                    h.update(block)
            if h.hexdigest() != sha:
                raise RuntimeError('Archive content mismatch: '+rel)
    sha = digest(ARCHIVE)
    Path(str(ARCHIVE) + '.sha256').write_text(sha + '  ' + ARCHIVE.name + '\n', encoding='utf-8')
    report = {'archive': ARCHIVE.name, 'archive_bytes': ARCHIVE.stat().st_size, 'archive_sha256': sha,
              'files': len(records)+1, 'unpacked_bytes': sum(r['bytes'] for r in records),
              'all_model_hashes_verified': True, 'zip_crc_verified': True,
              'personal_data_included': False, 'kind': 'Full standalone application package; external sensor/GPU/SteamVR installations required'}
    (REPORT / 'distribution-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
