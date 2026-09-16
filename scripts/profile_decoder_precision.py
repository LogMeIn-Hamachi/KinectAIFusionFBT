"""Development-only GPU kernel trace of the deployed TorchScript CUDA graph."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('mode',choices=['fp32','tf32'])
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/sam3d'/('decoder-kernels-'+args.mode)
out.mkdir(exist_ok=False)
torch.backends.cuda.matmul.allow_tf32=args.mode=='tf32'
torch.backends.cudnn.allow_tf32=args.mode=='tf32'
torch.set_num_threads(4)
torch._C._set_graph_executor_optimize(False)
with torch.inference_mode():
    model=torch.jit.load(str(root/'artifacts/sam3d/decoder-captured-v3/decoder.pt')).eval()
    folder=root/'artifacts/sam3d/export-reproducibility/decoder-inputs'
    shapes=[(1,1280,32,32),(1,2),(1,2),(1,3,3),(1,2)]
    tensors=[torch.from_numpy(np.fromfile(folder/f'input_{i}.bin',np.float32).reshape(shape)).cuda() for i,shape in enumerate(shapes)]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):model(*tensors)
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):outputs=model(*tensors)
        for _ in range(5):graph.replay()
        stream.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(3):graph.replay()
            stream.synchronize()
    prof.export_chrome_trace(str(out/'trace.json'))
    trace=json.loads((out/'trace.json').read_text())
    kernels={}
    for e in trace.get('traceEvents',[]):
        if e.get('cat')=='kernel':
            row=kernels.setdefault(e['name'],dict(count=0,total_us=0))
            row['count']+=1;row['total_us']+=e.get('dur',0)
    (out/'kernels.json').write_text(json.dumps(kernels,indent=2))
    print('GPU kernels:',len(kernels))
    for name,stats in sorted(kernels.items(),key=lambda x:-x[1]['total_us'])[:15]:print(stats,name)
