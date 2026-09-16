"""Move immutable traced index tensors to CUDA before graph capture.

TorchScript preserves Python list indices as CPU constants plus device copies.
Fold only these explicit, immutable constant copies; predictions are untouched.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch

def fold(module):
    graph=module.graph;count=0
    dtypes={4:torch.int64,6:torch.float32,7:torch.float64,11:torch.bool}
    for node in list(graph.nodes()):
        if node.kind() in ('aten::index_put_','aten::copy_'):
            slot=2 if node.kind()=='aten::index_put_' else 1
            value=node.inputsAt(slot).toIValue()
            if isinstance(value,torch.Tensor) and value.device.type=='cpu':
                with graph.insert_point_guard(node):replacement=graph.insertConstant(value.cuda())
                node.replaceInput(slot,replacement);count+=1
        if node.kind()!='aten::to' or not node.schema().startswith('aten::to.device'):continue
        inputs=list(node.inputs());value=inputs[0].toIValue();device=inputs[1].toIValue();dtype=inputs[2].toIValue()
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or str(device)!='cuda:0':continue
        if dtype not in dtypes:raise ValueError('Unexpected constant dtype')
        with graph.insert_point_guard(node):
            replacement=graph.insertConstant(value.to(device=device,dtype=dtypes[dtype]))
        node.output().replaceAllUsesWith(replacement);count+=1
    torch._C._jit_pass_dce(graph)
    return count

if __name__=='__main__':
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base=Path('artifacts/sam3d/decoder-torchscript-v4')
    out=Path('artifacts/sam3d/decoder-captured-v3');out.mkdir(exist_ok=False)
    module=torch.jit.load(str(base/'decoder.pt')).eval()
    count=fold(module);torch.jit.save(module,str(out/'decoder.pt'))
    shapes=[(1,1280,32,32),(1,2),(1,2),(1,3,3),(1,2)]
    inputs=tuple(torch.from_numpy(np.fromfile(base/'decoder-inputs'/f'input_{i}.bin',np.float32).reshape(shape)).cuda() for i,shape in enumerate(shapes))
    expected=np.load(base/'decoder-reference.npz')
    with torch.inference_mode():
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):module(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):result=module(*inputs)
        times=[]
        for _ in range(30):
            torch.cuda.synchronize();start=time.perf_counter();graph.replay();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
        errors=[]
        for a,b in zip(result,expected.values()):
            av=a.cpu().numpy();np.testing.assert_allclose(av,b,atol=1e-3,rtol=1e-5);errors.append(float(np.abs(av-b).max()))
    report={'constant_copies_folded':count,'median_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),'output_max_abs_errors':errors,'scope':'GPU-resident decoder only; Python development verification of native serialized module'}
    (out/'validation.json').write_text(json.dumps(report,indent=2));print(report)
