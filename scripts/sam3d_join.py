"""Join encoder and FP32 decoder, retaining features on the GPU in one session."""
import argparse
from pathlib import Path
import shutil
import onnx


def main():
    p=argparse.ArgumentParser()
    p.add_argument('encoder',type=Path);p.add_argument('decoder',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=False)
    enc=onnx.load(a.encoder,load_external_data=False)
    dec=onnx.load(a.decoder,load_external_data=False)
    enc=onnx.compose.add_prefix(enc,'encoder/')
    dec=onnx.compose.add_prefix(dec,'body/')
    assert {x.domain:x.version for x in enc.opset_import}=={x.domain:x.version for x in dec.opset_import}
    for n in dec.graph.node:
        for i,v in enumerate(n.input):
            if v=='body/features':n.input[i]='encoder/features'
    graph=onnx.helper.make_graph(list(enc.graph.node)+list(dec.graph.node),'FastSAM3DBody-landmarks',
          list(enc.graph.input)+[x for x in dec.graph.input if x.name!='body/features'],list(dec.graph.output),
          initializer=list(enc.graph.initializer)+list(dec.graph.initializer),
          value_info=list(enc.graph.value_info)+list(dec.graph.value_info))
    model=onnx.helper.make_model(graph,opset_imports=list(enc.opset_import),producer_name='KinectRGBD local conversion')
    model.ir_version=max(enc.ir_version,dec.ir_version)
    for t in onnx.external_data_helper._get_all_tensors(model):
        if t.data_location==onnx.TensorProto.EXTERNAL:
            fields={v.key:v.value for v in t.external_data}
            name=Path(fields['location'])
            if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe external location')
            dest=a.output.parent/name
            if not dest.exists():shutil.copyfile(a.encoder.parent/name,dest)
    onnx.save_model(model,a.output,save_as_external_data=True,all_tensors_to_one_file=True,
                    location='body.weights',size_threshold=1024)
    onnx.checker.check_model(str(a.output))
    print('Joined model:',a.output)


if __name__=='__main__':main()
