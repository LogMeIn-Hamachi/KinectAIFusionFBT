"""Specialize the serialized native decoder; preserve the existing runtime ABI."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sam-decoder-opt'
SOURCE=ROOT/'release/KinectSAM3D-Preview/assets/sam3d-fp8/decoder.pt'

@torch.jit.script
def parallel_sample(x:torch.Tensor,grid:torch.Tensor,interpolation:int,padding:int,align:bool):
    n,c,h,w=x.size(0),x.size(1),x.size(2),x.size(3)
    gh,gw=grid.size(1),grid.size(2)
    shared=grid.unsqueeze(1).expand(n,c,gh,gw,2).reshape(n*c,gh,gw,2)
    result=torch.grid_sampler(x.reshape(n*c,1,h,w),shared,interpolation,padding,align)
    return result.reshape(n,c,gh,gw)

def sampling(module):
    graph=module.graph;count=0
    for node in list(graph.nodes()):
        if node.kind()!='aten::grid_sampler':continue
        mapping=dict(zip(parallel_sample.graph.inputs(),node.inputs()))
        with graph.insert_point_guard(node):
            for inner in parallel_sample.graph.nodes():
                clone=graph.createClone(inner,lambda value:mapping[value]);graph.insertNode(clone)
                mapping.update(zip(inner.outputs(),clone.outputs()))
        node.output().replaceAllUsesWith(mapping[next(parallel_sample.graph.outputs())]);node.destroy();count+=1
    torch._C._jit_pass_dce(graph);graph.lint()
    assert count==5
    return count

def precision(module,dtype):
    graph=module.graph;count=0;constants={}
    for node in list(graph.nodes()):
        if node.kind()!='aten::linear':continue
        weight=node.inputsAt(1).toIValue()
        if not isinstance(weight,torch.Tensor) or tuple(weight.shape)[0] not in (512,1024,2048) or tuple(weight.shape)[1] not in (512,1024,1280,2048):continue
        with graph.insert_point_guard(node):
            typ=graph.insertConstant(5 if dtype==torch.float16 else 15)
            fp32=graph.insertConstant(6);no=graph.insertConstant(False);none=graph.insertConstant(None)
            cast=graph.create('aten::to',[node.inputsAt(0),typ,no,no,none]);cast.output().setType(torch._C.TensorType.get());graph.insertNode(cast)
            node.replaceInput(0,cast.output())
            for slot in (1,2):
                value=node.inputsAt(slot);tensor=value.toIValue()
                if isinstance(tensor,torch.Tensor):
                    if value not in constants:constants[value]=graph.insertConstant(tensor.to(dtype))
                    node.replaceInput(slot,constants[value])
        castback=graph.create('aten::to',[node.output(),fp32,no,no,none]);castback.output().setType(torch._C.TensorType.get());castback.insertAfter(node)
        node.output().replaceAllUsesAfterNodeWith(castback,castback.output());count+=1
    torch._C._jit_pass_dce(graph);graph.lint()
    return count

def check_sampling():
    torch.manual_seed(923);cases=0;maximum=0.
    with torch.inference_mode():
        for n,c,gh,gw in ((1,1280,70,1),(1,1280,1,1),(2,32,7,3)):
            x=torch.randn(n,c,32,32,device='cuda')
            grid=torch.rand(n,gh,gw,2,device='cuda')*3-1.5
            grid.flatten()[:4]=torch.tensor([-1,1,-1.001,1.001],device='cuda')[:min(grid.numel(),4)]
            for mode in (0,1,2):
                for padding in (0,1,2):
                    for align in (False,True):
                        a=torch.grid_sampler(x,grid,mode,padding,align);b=parallel_sample(x,grid,mode,padding,align)
                        torch.testing.assert_close(a,b,rtol=1e-6,atol=2e-6);cases+=1
                        maximum=max(maximum,float((a-b).abs().max()))
    report=dict(cases=cases,max_abs_difference=maximum,scope='Unit-scale synthetic values; 3 layouts, all interpolation/padding/align modes; permits float32 arithmetic roundoff.')
    (OUT/'sampling-checks.json').write_text(json.dumps(report,indent=2))
    return report

def export():
    OUT.mkdir(exist_ok=True)
    print('Sampling edge/layout checks:',check_sampling(),flush=True)
    for name,dtype in (('sampling',None),('sampling-fp16',torch.float16),('sampling-bf16',torch.bfloat16)):
        module=torch.jit.load(str(SOURCE)).eval()
        with SOURCE.open('rb') as f:source_hash=hashlib.file_digest(f,'sha256').hexdigest()
        report={'source_decoder_sha256':source_hash,'sampling_nodes':sampling(module),'low_precision_linears':precision(module,dtype) if dtype else 0}
        folder=OUT/name;folder.mkdir(exist_ok=True)
        torch.jit.save(module,str(folder/'decoder.pt'))
        (folder/'conversion.json').write_text(json.dumps(report,indent=2));print(name,report,flush=True)

def bench(path):
    torch.set_num_threads(4);torch._C._set_graph_executor_optimize(False)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    module=torch.jit.load(str(path)).eval()
    folder=ROOT/'artifacts/sam3d/export-reproducibility/decoder-inputs'
    shapes=[(1,1280,32,32),(1,2),(1,2),(1,3,3),(1,2)]
    with torch.inference_mode():
        inputs=[torch.from_numpy(np.fromfile(folder/f'input_{i}.bin',np.float32).reshape(shape)).cuda() for i,shape in enumerate(shapes)]
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):module(*inputs)
            stream.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):output=module(*inputs)
            times=[]
            for _ in range(50):
                stream.synchronize();start=time.perf_counter();graph.replay();stream.synchronize();times.append((time.perf_counter()-start)*1000)
        result=dict(median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)))
        folder=path.parent
        if path==SOURCE:folder=OUT/'baseline';folder.mkdir(exist_ok=True)
        for i,o in enumerate(output):o.cpu().numpy().tofile(folder/f'bench-output-{i}.bin')
        (folder/'bench.json').write_text(json.dumps(result,indent=2));print(result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['export','bench']);p.add_argument('--model',type=Path)
    a=p.parse_args()
    if a.command=='export':export()
    else:bench(a.model.resolve() if a.model else SOURCE)
