"""Partition at decoder refinement boundaries to bound TensorRT compiler work.

No weights or operations are removed. Cross-boundary tensors retain names,
types, shapes and values. This is an alternative deployment experiment.
"""
from pathlib import Path
import json
import onnx

base=Path('artifacts/sam3d')
out=base/'decoder-partitioned';out.mkdir(exist_ok=False)
m=onnx.shape_inference.infer_shapes(onnx.load(base/'decoder-landmark-v8/decoder-optimized.onnx'))
nodes=list(m.graph.node)
cuts=[0]+[i for i,n in enumerate(nodes) if n.name in
      [f'/decoder/layers.{j}/ln1/LayerNormalization' for j in [1,2,3]]]+[len(nodes)]
types={x.name:x for x in list(m.graph.input)+list(m.graph.value_info)+list(m.graph.output)}
weights={x.name:x for x in m.graph.initializer}
parts=[]
for part,(a,b) in enumerate(zip(cuts,cuts[1:])):
    ns=nodes[a:b];produced={x for n in ns for x in n.output};used={x for n in ns for x in n.input if x}
    later={x for n in nodes[b:] for x in n.input}|{x.name for x in m.graph.output}
    inputs=sorted(used-produced-weights.keys());outputs=sorted(produced&later)
    if part==0:inputs=[x.name for x in m.graph.input if x.name in inputs]
    for name in inputs+outputs:
        if name not in types:raise ValueError('Missing tensor shape: '+name)
    g=onnx.helper.make_graph(ns,f'decoder_stage_{part}',[types[x] for x in inputs],[types[x] for x in outputs],
          initializer=[weights[x] for x in sorted(used&weights.keys())])
    model=onnx.helper.make_model(g,opset_imports=list(m.opset_import));model.ir_version=m.ir_version
    path=out/f'stage{part}.onnx';onnx.save(model,path);onnx.checker.check_model(model)
    parts.append({'file':path.name,'inputs':inputs,'outputs':outputs,'nodes':len(ns)})
(out/'pipeline.json').write_text(json.dumps(parts,indent=2))
print(json.dumps(parts,indent=2))
