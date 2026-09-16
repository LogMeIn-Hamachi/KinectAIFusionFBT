"""Compare the native decoder precision modes on identical saved encoder outputs."""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/sam3d/decoder-precision-validation'

def difference(a, b):
    xyz = np.linalg.norm(a[0].reshape(-1,3)-b[0].reshape(-1,3),axis=-1)*1000
    uv = np.linalg.norm(a[1].reshape(-1,2)-b[1].reshape(-1,2),axis=-1)
    ra,rb = [x[2].reshape(-1,3,3).astype(np.float64) for x in (a,b)]
    # Relative angle from atan2 is stable around zero despite FP32 matrix drift.
    r = ra @ rb.transpose(0,2,1)
    sine = np.linalg.norm(np.stack((r[:,2,1]-r[:,1,2],r[:,0,2]-r[:,2,0],r[:,1,0]-r[:,0,1]),axis=-1),axis=-1)/2
    angle = np.rad2deg(np.arctan2(sine,(np.trace(r,axis1=1,axis2=2)-1)/2))
    return dict(mean_joint_mm=float(xyz.mean()),max_joint_mm=float(xyz.max()),max_image_px=float(uv.max()),max_rig_rotation_deg=float(angle.max()))

def main():
    OUT.mkdir(exist_ok=False)
    rows=[]
    for frame in range(0,301,30):
        folder=OUT/str(frame);folder.mkdir()
        inputs=folder/'inputs';inputs.mkdir()
        shutil.copyfile(ROOT/f'artifacts/sam3d/encoder-all-native/{frame}-inputs/output_0.bin',inputs/'input_0.bin')
        for i in (1,2,3):
            shutil.copyfile(ROOT/f'artifacts/sam3d/reference-all/{frame}-inputs/input_{i}.bin',inputs/f'input_{i}.bin')
        outputs={};times={}
        # Alternate order to reduce systematic warm-clock bias.
        for mode in (('fp32','tf32') if frame%60==0 else ('tf32','fp32')):
            dest=folder/mode
            subprocess.run([str(ROOT/'build/Release/kf_torch_bench.exe'),str(ROOT/'release/KinectSAM3D-Preview/sam3d-runtime'),str(ROOT/'artifacts/sam3d/decoder-captured-v3/decoder.pt'),str(inputs),str(dest),mode,str(ROOT/'build/Release/kf_sam3d_decoder.dll')],check=True,cwd=ROOT)
            outputs[mode]=[np.fromfile(dest/f'output_{i}.bin',np.float32) for i in range(3)]
            assert all(np.isfinite(x).all() for x in outputs[mode])
            times[mode]=json.loads((dest/'result.json').read_text())
        reference=np.load(ROOT/f'artifacts/sam3d/reference-all/{frame}.npz')
        ref=[reference[k].flatten() for k in ('camera_points','image_points','joint_rotations_mhr')]
        row=dict(frame=frame,tf32_vs_native_fp32=difference(outputs['tf32'],outputs['fp32']),tf32_vs_original_full_fp32=difference(outputs['tf32'],ref),timings=times)
        rows.append(row)
        print(json.dumps(row),flush=True)
    report=dict(scope='11 recorded inputs; identical saved TensorRT encoder features. Native decoder timing includes host transfers. Numerical fidelity, not ground truth tracking accuracy.',samples=rows)
    report['summary']={mode+'_median_ms':float(np.median([r['timings'][mode]['median_ms'] for r in rows])) for mode in ('fp32','tf32')}
    report['summary'].update({k:max(r['tf32_vs_native_fp32'][k] for r in rows) for k in ('max_joint_mm','max_image_px','max_rig_rotation_deg')})
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary']),flush=True)

if __name__=='__main__':main()
