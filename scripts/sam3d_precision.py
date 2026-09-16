"""Convert the image encoder to mixed FP16; leave the body/rig graph FP32."""
import argparse
from pathlib import Path
import onnx
from onnxconverter_common import float16

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    if args.output.exists():raise RuntimeError('Output already exists')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    print('Loading FP32 encoder',flush=True)
    model=onnx.load(args.input)
    # Strongly typed TensorRT reads precision from the graph. Keep numerically
    # sensitive reductions and normalization in FP32; do not quantize weights.
    blocked=list(set(float16.DEFAULT_OP_BLOCK_LIST+['LayerNormalization','Softmax','ReduceMean','ReduceSum','Pow','Sqrt','Div']))
    model=float16.convert_float_to_float16(model,keep_io_types=True,op_block_list=blocked,
                                          min_positive_val=1e-8,max_finite_val=65504,disable_shape_infer=True)
    for t in onnx.external_data_helper._get_all_tensors(model):
        if t.HasField('raw_data'):
            t.ClearField('external_data');t.data_location=onnx.TensorProto.DEFAULT
    onnx.save_model(model,args.output,save_as_external_data=True,all_tensors_to_one_file=True,
                    location=args.output.name+'.data',size_threshold=1024)
    print('Saved mixed FP16 encoder; accuracy not yet accepted',flush=True)

if __name__=='__main__':main()
