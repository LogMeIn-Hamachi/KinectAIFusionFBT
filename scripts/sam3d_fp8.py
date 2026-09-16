"""Development-only selective E4M3 encoder quantization and NVIDIA validation.

Explicit Q/DQ, scalar max calibration. Does not change installed model assets.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import numpy as np
import ml_dtypes
import onnx
from onnx import helper as H, numpy_helper as N, TensorProto as T
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sam-fp8'
LIBS=ROOT/'release/KinectSAM3D-Preview'
SOURCE=LIBS/'assets/sam3d/backbone.onnx'

def runtime(path):
    ort.register_execution_provider_library('NvTensorRTRTXExecutionProvider',str(LIBS/'onnxruntime_providers_nv_tensorrt_rtx.dll'))
    devices=[d for d in ort.get_ep_devices() if d.ep_name=='NvTensorRTRTXExecutionProvider'][:1]
    opts=ort.SessionOptions();opts.intra_op_num_threads=4;opts.inter_op_num_threads=1
    opts.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    opts.add_session_config_entry('session.intra_op.allow_spinning','0')
    key=hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    cache=OUT/'cache'/key;cache.mkdir(parents=True,exist_ok=True)
    opts.add_provider_for_devices(devices,{'nv_max_workspace_size':'536870912','enable_cuda_graph':'1','nv_runtime_cache_path':str(cache)})
    return ort.InferenceSession(str(path),sess_options=opts)

def zero(name):return H.make_tensor(name,T.FLOAT8E4M3FN,[],b'\0',raw=True)
def scalar(name,value):return N.from_array(np.array(value,dtype=np.float16),name)

def probe():
    rng=np.random.default_rng(1)
    w=rng.normal(0,.1,(128,128)).astype(np.float16);s=np.float16(np.abs(w).max()/448)
    q=N.from_array((w/s).astype(ml_dtypes.float8_e4m3fn),'w8')
    nodes=[H.make_node('QuantizeLinear',['x','as','az'],['x8']),H.make_node('DequantizeLinear',['x8','as','az'],['xd']),H.make_node('DequantizeLinear',['w8','ws','wz'],['wd']),H.make_node('MatMul',['xd','wd'],['y'])]
    graph=H.make_graph(nodes,'fp8_probe',[H.make_tensor_value_info('x',T.FLOAT16,[1,128,128])],[H.make_tensor_value_info('y',T.FLOAT16,[1,128,128])],[q,scalar('as',.01),scalar('ws',s),zero('az'),zero('wz')])
    m=H.make_model(graph,opset_imports=[H.make_opsetid('',21)]);m.ir_version=10
    path=OUT/'probe.onnx';onnx.save(m,path);session=runtime(path)
    x=rng.normal(0,1,(1,128,128)).astype(np.float16)
    y=session.run(None,{'x':x})[0]
    assert np.isfinite(y).all()
    print('NVIDIA FP8 Q/DQ matrix execution passed; max probe difference:',float(np.max(abs(y.astype(np.float32)-x.astype(np.float32)@w.astype(np.float32)))),flush=True)

def selected(m,first=1,last=31):
    init={i.name for i in m.graph.initializer}
    linears=[n for n in m.graph.node if n.op_type=='MatMul' and n.input[1] in init]
    assert len(linears)==160
    return [n for i,n in enumerate(linears) if first<=i//5<last and i%5>=2]

def calibrate():
    m=onnx.load(SOURCE,load_external_data=False)
    data=OUT/'backbone.onnx.data'
    if not data.exists():shutil.copyfile(SOURCE.parent/'backbone.onnx.data',data)
    inputs=sorted({n.input[0] for n in selected(m,0,32)})
    # Only small reduction results return to the CPU, not all hidden activations.
    for i,name in enumerate(inputs):
        a=f'fp8_abs_{i}';r=f'fp8_max_{i}'
        m.graph.node.extend([H.make_node('Abs',[name],[a]),H.make_node('ReduceMax',[a],[r],keepdims=0)])
        m.graph.output.append(H.make_tensor_value_info(r,T.FLOAT16,[]))
    path=OUT/'calibration.onnx';onnx.save(m,path)
    print('Compiling instrumented encoder for calibration',flush=True)
    session=runtime(path);maxima=np.zeros(len(inputs))
    frames=list(range(0,301,60))
    for frame in frames:
        x=np.fromfile(ROOT/f'artifacts/sam3d/reference-all/{frame}-inputs/input_0.bin',np.float32).reshape(1,3,512,512)
        values=session.run(None,{'image':x})
        maxima=np.maximum(maxima,np.array([float(v) for v in values[1:]]))
        assert np.isfinite(maxima).all()
        print('Calibrated frame',frame,flush=True)
    report=dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),frames=frames,method='Per-tensor maximum absolute value, 10% headroom, E4M3FN',maxima=dict(zip(inputs,maxima.tolist())))
    (OUT/'calibration.json').write_text(json.dumps(report,indent=2))

def convert(first,last):
    report=json.loads((OUT/'calibration.json').read_text())
    assert report['source_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    m=onnx.load(SOURCE);targets={n.name for n in selected(m,first,last)}
    init={i.name:i for i in m.graph.initializer};nodes=[];extra=[];replaced=set();quantized={}
    for node in m.graph.node:
        if node.name in targets:
            a,w=node.input
            if a not in quantized:
                base=a+'_fp8';scale=max(report['maxima'][a]*1.1/448,1e-7)
                extra.extend([scalar(base+'_s',scale),zero(base+'_z')])
                nodes.extend([H.make_node('QuantizeLinear',[a,base+'_s',base+'_z'],[base+'_q']),H.make_node('DequantizeLinear',[base+'_q',base+'_s',base+'_z'],[base+'_dq'])])
                quantized[a]=base+'_dq'
            values=N.to_array(init[w]).astype(np.float32);scale=max(float(np.abs(values).max())/448,1e-7)
            scale=np.float16(scale)
            q=np.clip(values/scale,-448,448).astype(ml_dtypes.float8_e4m3fn)
            base=w+'_fp8'
            extra.extend([N.from_array(q,base+'_q'),scalar(base+'_s',scale),zero(base+'_z')])
            nodes.append(H.make_node('DequantizeLinear',[base+'_q',base+'_s',base+'_z'],[base+'_dq']))
            node.input[0]=quantized[a];node.input[1]=base+'_dq';replaced.add(w)
        nodes.append(node)
    del m.graph.node[:];m.graph.node.extend(nodes)
    # No other consumers of the replaced constant matrices are permitted.
    assert not any(x in replaced for n in nodes for x in n.input)
    constants=[i for i in m.graph.initializer if i.name not in replaced]
    del m.graph.initializer[:];m.graph.initializer.extend(constants+extra)
    values=[v for v in m.graph.value_info if v.name not in replaced]
    del m.graph.value_info[:];m.graph.value_info.extend(values)
    for op in m.opset_import:
        if not op.domain:op.version=21
    folder=OUT/f'mlp-{first}-{last}';folder.mkdir(exist_ok=True)
    path=folder/'backbone.onnx'
    if path.exists():raise RuntimeError('Candidate exists')
    for t in onnx.external_data_helper._get_all_tensors(m):
        if t.HasField('raw_data'):t.ClearField('external_data');t.data_location=T.DEFAULT
    onnx.save_model(m,path,save_as_external_data=True,all_tensors_to_one_file=True,location='backbone.onnx.data',size_threshold=1024)
    onnx.checker.check_model(str(path))
    (folder/'conversion.json').write_text(json.dumps(dict(first_block=first,last_block_exclusive=last,quantized_linears=sorted(targets),activation_scales=report,source='Existing mixed-FP16 DINOv3 encoder; attention, normalization, first/last excluded blocks and decoder unchanged'),indent=2))
    print('Saved',len(targets),'FP8 linear layers to',path,flush=True)

def compare(path):
    start=time.perf_counter();session=runtime(path)
    print('Loaded in',time.perf_counter()-start,'s',flush=True)
    folder=path.parent/'outputs';folder.mkdir(exist_ok=True);rows=[]
    for frame in range(0,301,30):
        x=np.fromfile(ROOT/f'artifacts/sam3d/reference-all/{frame}-inputs/input_0.bin',np.float32).reshape(1,3,512,512)
        for _ in range(3):y=session.run(None,{'image':x})[0]
        times=[]
        for _ in range(10):
            start=time.perf_counter();y=session.run(None,{'image':x})[0];times.append((time.perf_counter()-start)*1000)
        assert np.isfinite(y).all();y.tofile(folder/f'{frame}.bin')
        ref=np.fromfile(ROOT/f'artifacts/sam3d/encoder-all-native/{frame}-inputs/output_0.bin',np.float32).reshape(y.shape)
        row=dict(frame=frame,held_out=frame%60!=0,median_ms=float(np.median(times)),feature_relative_rmse=float(np.sqrt(np.mean((y-ref)**2)/np.mean(ref**2))))
        rows.append(row);print(row,flush=True)
    (path.parent/'encoder-comparison.json').write_text(json.dumps(rows,indent=2))

def profile(path):
    print('Importing profiling runtime',flush=True)
    import torch
    print('Creating encoder profiling session',flush=True)
    session=runtime(path)
    x=np.fromfile(ROOT/'artifacts/sam3d/reference-all/150-inputs/input_0.bin',np.float32).reshape(1,3,512,512)
    for _ in range(5):session.run(None,{'image':x})
    print('Capturing three GPU runs',flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(3):session.run(None,{'image':x})
        torch.cuda.synchronize()
    trace=path.parent/'gpu-trace.json';prof.export_chrome_trace(str(trace))
    kernels={}
    for e in json.loads(trace.read_text()).get('traceEvents',[]):
        if e.get('cat')=='kernel':
            v=kernels.setdefault(e['name'],dict(count=0,total_us=0));v['count']+=1;v['total_us']+=e.get('dur',0)
    (path.parent/'gpu-kernels.json').write_text(json.dumps(kernels,indent=2))
    for name,v in sorted(kernels.items(),key=lambda x:-x[1]['total_us'])[:8]:print(v,name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['probe','calibrate','convert','compare','profile']);p.add_argument('--first',type=int,default=1);p.add_argument('--last',type=int,default=31);p.add_argument('--model',type=Path)
    a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.command=='probe':probe()
    elif a.command=='calibrate':calibrate()
    elif a.command=='convert':convert(a.first,a.last)
    elif a.command=='profile':profile(a.model.resolve())
    else:compare(a.model.resolve())
