"""Check explicit image dimensions against TorchScript and the retained v1 ABI."""
from pathlib import Path
import ctypes as C
import json
import os
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
libs=ROOT/'release/KinectSAM3D-Preview/sam3d-runtime'
directory=os.add_dll_directory(str(libs))
torch.cuda.init()
dll=C.CDLL(str(libs/'kf_sam3d_decoder.dll'))
P=C.POINTER(C.c_float)
dll.kf_decoder_create_precision.argtypes=[C.c_char_p,C.c_int]
dll.kf_decoder_create_precision.restype=C.c_void_p
dll.kf_decoder_destroy.argtypes=[C.c_void_p]
dll.kf_decoder_error.restype=C.c_char_p
dll.kf_decoder_feature_buffer.argtypes=[C.c_void_p,C.POINTER(P),C.POINTER(C.c_size_t),C.POINTER(C.c_int)]
dll.kf_decoder_run.argtypes=[C.c_void_p]+[P]*7
dll.kf_decoder_run_gpu_size.argtypes=[C.c_void_p]+[P]*8
def ptr(a):return a.ctypes.data_as(P)
def require(ok):
    if not ok:raise RuntimeError(dll.kf_decoder_error().decode())

model=ROOT/'release/KinectSAM3D-Preview/assets/sam3d-optimized/decoder.pt'
handle=dll.kf_decoder_create_precision(str(model).encode(),1)
require(handle)
results=[]
try:
    device=P();count=C.c_size_t();index=C.c_int()
    require(dll.kf_decoder_feature_buffer(handle,C.byref(device),C.byref(count),C.byref(index)))
    reference=torch.jit.load(str(model),map_location='cuda').eval()
    with torch.inference_mode(),torch.jit.optimized_execution(False):
        for frame in (0,90,180,300):
            features=np.fromfile(ROOT/f'artifacts/sam3d/encoder-all-native/{frame}-inputs/output_0.bin',np.float32)
            small=[np.fromfile(ROOT/f'artifacts/sam3d/reference-all/{frame}-inputs/input_{i}.bin',np.float32) for i in (1,2,3)]
            output=[np.zeros(n,np.float32) for n in (210,140,1143)]
            require(dll.kf_decoder_run(handle,ptr(features),*[ptr(a) for a in small+output]))
            legacy=[a.copy() for a in output]
            wh=np.array([640,480],np.float32)
            require(dll.kf_decoder_run_gpu_size(handle,device,*[ptr(a) for a in small],ptr(wh),*[ptr(a) for a in output]))
            assert all(np.array_equal(a,b) for a,b in zip(legacy,output)), 'V1 ABI changed'
            center,scale,camera=[a.copy() for a in small]
            center*=np.array([3,2.25],np.float32);scale*=2.25
            camera=camera.reshape(3,3);camera[0]*=3;camera[1]*=2.25
            wh=np.array([1920,1080],np.float32)
            new=[center,scale,camera]
            require(dll.kf_decoder_run_gpu_size(handle,device,*[ptr(a) for a in new],ptr(wh),*[ptr(a) for a in output]))
            args=[torch.from_numpy(a.copy()).cuda().reshape(shape) for a,shape in zip([features]+new+[wh],[(1,1280,32,32),(1,2),(1,2),(1,3,3),(1,2)])]
            expected=[v.cpu().numpy().reshape(-1) for v in reference(*args)]
            errors=[float(np.max(np.abs(a-b))) for a,b in zip(output,expected)]
            for a,b in zip(output,expected):np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-5)
            for invalid in ([0,1080],[np.nan,1080],[1920,np.inf],[8192,1080]):
                assert not dll.kf_decoder_run_gpu_size(handle,device,*[ptr(a) for a in new],ptr(np.array(invalid,np.float32)),*[ptr(a) for a in output])
            require(dll.kf_decoder_run_gpu_size(handle,device,*[ptr(a) for a in new],ptr(wh),*[ptr(a) for a in output]))
            results.append(dict(frame=frame,v1_exact=True,v2_max_abs_errors_xyz_uv_rot=errors,invalid_dimensions_rejected=True,recovered=True))
finally:dll.kf_decoder_destroy(handle)
report=dict(scope='Four saved feature sets; full-HD native CUDA Graph versus TorchScript with explicit dimensions; no live capture',cases=results)
(ROOT/'artifacts/kinect-v2/decoder-validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
