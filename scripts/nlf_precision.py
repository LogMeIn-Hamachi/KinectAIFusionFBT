"""NVIDIA graph precision audit, including an intentionally mostly offscreen crop."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import onnx
from onnxconverter_common import float16
import onnxruntime as ort
import torch
import torchvision

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nlf-validation'
libs=ROOT/'release/KinectSAM3D-Preview'
torch.set_num_threads(4)
ort.register_execution_provider_library('NvTensorRTRTXExecutionProvider',str(libs/'onnxruntime_providers_nv_tensorrt_rtx.dll'))
devices=[d for d in ort.get_ep_devices() if d.ep_name=='NvTensorRTRTXExecutionProvider'][:1]
m=torch.jit.load(str(ROOT/'assets/nlf/source/nlf_s_multi_0.2.2.torchscript'),map_location='cpu').float().eval()
base=ROOT/'assets/nlf'
graph=onnx.load(base/'pose-fp32.onnx')
for n in graph.graph.node:
    if n.op_type=='Cast' and any(a.name=='to' and a.i==onnx.TensorProto.FLOAT for a in n.attribute):
        n.op_type='Identity';del n.attribute[:]
blocked=list(set(float16.DEFAULT_OP_BLOCK_LIST+['Softmax','ReduceMean','ReduceSum','Pow','Sqrt','Div','Softplus']))
mixed=float16.convert_float_to_float16(graph,keep_io_types=True,op_block_list=blocked,
    node_block_list=['/Conv_170','/Conv_171'],min_positive_val=1e-8)
onnx.save(mixed,base/'pose-conservative.onnx')
reports={}
for name in (['pose.onnx'] if '--default-only' in sys.argv else ['pose-fp16.onnx','pose-conservative.onnx','pose-fp32.onnx']):
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    options.add_provider_for_devices(devices,{'nv_max_workspace_size':'134217728','enable_cuda_graph':'1'})
    session=ort.InferenceSession(str(base/name),sess_options=options)
    rows=[]
    with torch.no_grad():
        for i in range(11):
            prefix=OUT/str(i)
            image=np.fromfile(str(prefix)+'.crop.bin',np.float32).reshape(1,3,256,256)
            output=torch.from_numpy(session.run(None,{'image':image})[0])
            warp=np.loadtxt(str(prefix)+'.warp.txt').reshape(3,3,3)
            R=torch.tensor(warp[0],dtype=torch.float32);K=torch.tensor(warp[1],dtype=torch.float32)[None]
            decoded,_=m.crop_model.heatmap_head.reconstruct_absolute(output[:,:,:2],output[:,:,2:5],output[:,:,5],K)
            actual=((decoded/1000)@R).numpy()[0]
            reference=np.fromfile(str(prefix)+'.reference3d.bin',np.float32).reshape(26,3)
            error=np.linalg.norm(actual-reference,axis=1)*1000
            rows.append(dict(sample=i,max_mm=float(error.max()),mean_mm=float(error.mean())))
    # Same ONNX Runtime, plugin and model as the native executable; Python is only
    # the profiling harness. Capture actual CUDA kernel launches, not utilization.
    for _ in range(5):session.run(None,{'image':image})
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):session.run(None,{'image':image})
        torch.cuda.synchronize()
    trace=OUT/(name+('-strict' if os.getenv('NVIDIA_TF32_OVERRIDE')=='0' else '-default')+'-trace.json');prof.export_chrome_trace(str(trace))
    kernels={}
    for e in json.loads(trace.read_text()).get('traceEvents',[]):
        if e.get('cat')=='kernel':
            item=kernels.setdefault(e['name'],dict(count=0,total_us=0));item['count']+=1;item['total_us']+=e.get('dur',0)
    reports[name]=dict(numeric=rows,kernels=kernels)
    print(name,rows,flush=True)
    print('Kernels',sorted(kernels.items(),key=lambda x:-x[1]['total_us'])[:6],flush=True)
    del session
(OUT/('precision-strict.json' if os.getenv('NVIDIA_TF32_OVERRIDE')=='0' else 'precision-current.json')).write_text(json.dumps(reports,indent=2))
