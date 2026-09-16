"""Decode saved FP8 and reference features through the unchanged native SAM DLL."""
from pathlib import Path
import argparse
import ctypes as C
import json
import os
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);a=p.parse_args()
folder=a.candidate.resolve()
libs=ROOT/'release/KinectSAM3D-Preview/sam3d-runtime'
directory=os.add_dll_directory(str(libs));torch.cuda.init()
dll=C.CDLL(str(libs/'kf_sam3d_decoder.dll'));P=C.POINTER(C.c_float)
dll.kf_decoder_create_precision.argtypes=[C.c_char_p,C.c_int];dll.kf_decoder_create_precision.restype=C.c_void_p
dll.kf_decoder_destroy.argtypes=[C.c_void_p];dll.kf_decoder_destroy.restype=None
dll.kf_decoder_error.restype=C.c_char_p
dll.kf_decoder_run.argtypes=[C.c_void_p]+[P]*7
handle=dll.kf_decoder_create_precision(str(ROOT/'release/KinectSAM3D-Preview/assets/sam3d/decoder.pt').encode(),1)
if not handle:raise RuntimeError(dll.kf_decoder_error().decode())

def decode(features,inputs):
    output=[np.zeros(n,np.float32) for n in (210,140,1143)]
    args=[x.ctypes.data_as(P) for x in [features]+inputs+output]
    if not dll.kf_decoder_run(handle,*args):raise RuntimeError(dll.kf_decoder_error().decode())
    assert all(np.isfinite(x).all() for x in output)
    return output[0].reshape(70,3).astype(np.float64)

rows=[]
try:
    for frame in range(0,301,30):
        inputs=[np.fromfile(ROOT/f'artifacts/sam3d/reference-all/{frame}-inputs/input_{i}.bin',np.float32) for i in (1,2,3)]
        ref=decode(np.fromfile(ROOT/f'artifacts/sam3d/encoder-all-native/{frame}-inputs/output_0.bin',np.float32),inputs)
        fp8=decode(np.fromfile(folder/f'outputs/{frame}.bin',np.float32),inputs)
        center=lambda x:(x[9]+x[10])/2
        delta=np.linalg.norm(fp8-ref,axis=1)*1000
        local=np.linalg.norm((fp8-center(fp8))-(ref-center(ref)),axis=1)*1000
        body=list(range(21))+[41,62,69]
        row=dict(frame=frame,held_out=frame%60!=0,max_camera_point_mm=float(delta.max()),max_body_local_mm=float(local[body].max()),p95_body_local_mm=float(np.percentile(local[body],95)),max_feet_local_mm=float(local[13:21].max()))
        rows.append(row);print(row,flush=True)
finally:dll.kf_decoder_destroy(handle)
report=dict(scope='Unchanged native TF32 decoder; fidelity to current SAM, not ground truth. Six calibration frames, five held-out frames.',frames=rows)
(folder/'pose-comparison.json').write_text(json.dumps(report,indent=2))
