"""Exercise the native GPU buffer ABI, legacy compatibility and rejection paths.

Development harness only. Uses saved encoder features; no capture or OSC.
"""
from pathlib import Path
import ctypes as C
import json
import os
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
libs=ROOT/'release/KinectSAM3D-Preview/sam3d-runtime'
directory=os.add_dll_directory(str(libs))
# Initialize the development interpreter's matching CUDA dependencies first.
# The shipped native application does not import or depend on Python.
torch.cuda.init()
dll=C.CDLL(str(libs/'kf_sam3d_decoder.dll'))
cuda=C.CDLL(str(libs/'cudart64_12.dll'))
P=C.POINTER(C.c_float)
dll.kf_decoder_create_precision.argtypes=[C.c_char_p,C.c_int]
dll.kf_decoder_create_precision.restype=C.c_void_p
dll.kf_decoder_destroy.argtypes=[C.c_void_p]
dll.kf_decoder_destroy.restype=None
dll.kf_decoder_error.restype=C.c_char_p
dll.kf_decoder_feature_buffer.argtypes=[C.c_void_p,C.POINTER(P),C.POINTER(C.c_size_t),C.POINTER(C.c_int)]
for name in ('kf_decoder_run','kf_decoder_run_gpu'):
    getattr(dll,name).argtypes=[C.c_void_p]+[P]*7
cuda.cudaMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
cuda.cudaMemcpy.restype=C.c_int

def pointer(a):return a.ctypes.data_as(P)
def require(result):
    if not result:raise RuntimeError(dll.kf_decoder_error().decode())

results=[]
for precision in (0,1):
    handle=dll.kf_decoder_create_precision(str(ROOT/'release/KinectSAM3D-Preview/assets/sam3d/decoder.pt').encode(),precision)
    require(handle)
    try:
        device=P();count=C.c_size_t();index=C.c_int()
        require(dll.kf_decoder_feature_buffer(handle,C.byref(device),C.byref(count),C.byref(index)))
        assert count.value==1280*32*32 and index.value>=0
        for frame in range(0,301,30):
            features=np.fromfile(ROOT/f'artifacts/sam3d/encoder-all-native/{frame}-inputs/output_0.bin',np.float32)
            inputs=[np.fromfile(ROOT/f'artifacts/sam3d/reference-all/{frame}-inputs/input_{i}.bin',np.float32) for i in (1,2,3)]
            output=[np.zeros(n,np.float32) for n in (210,140,1143)]
            args=[pointer(a) for a in inputs+output]
            require(dll.kf_decoder_run(handle,pointer(features),*args))
            reference=[a.copy() for a in output]
            # Legacy call populated the very same persistent GPU buffer.
            require(dll.kf_decoder_run_gpu(handle,device,*args))
            assert all(np.array_equal(a,b) for a,b in zip(reference,output))
            assert not dll.kf_decoder_run_gpu(handle,pointer(features),*args)
            assert b'Wrong SAM GPU feature buffer' in dll.kf_decoder_error()
            for bad in (np.nan,np.inf,-np.inf):
                value=np.array([bad],np.float32)
                assert cuda.cudaMemcpy(device,pointer(value),4,1)==0
                assert not dll.kf_decoder_run_gpu(handle,device,*args)
                assert b'Nonfinite SAM image features' in dll.kf_decoder_error()
            # Restore data using the legacy ABI and verify recovery.
            require(dll.kf_decoder_run(handle,pointer(features),*args))
            require(dll.kf_decoder_run_gpu(handle,device,*args))
            assert all(np.array_equal(a,b) for a,b in zip(reference,output))
            results.append(dict(frame=frame,precision='tf32' if precision else 'fp32',exact_output_match=True,rejected_wrong_pointer=True,rejected_nan_and_infinities=True,recovered=True))
    finally:dll.kf_decoder_destroy(handle)
report=dict(scope='Native DLL ABI on 11 saved feature inputs in each precision; no sensor/OSC',cases=results)
(ROOT/'artifacts/sam-gpu-transfer/buffer-validation.json').write_text(json.dumps(report,indent=2))
print('22 native GPU/host comparisons exact; wrong pointer and NaN/infinities rejected; recovery passed.')
