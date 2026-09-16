"""Specialize official NLF-S to 26 body/foot points for NVIDIA TensorRT.

Development only. No detector, mesh fitting, field evaluation, or Python at runtime.
Network returns crop UV, relative metric XYZ and uncertainty; native geometry
implements the upstream regularized perspective reconstruction in double precision.
Default: FP32 graph with runtime TF32 acceleration; FP16 is an audit candidate only.
"""
from pathlib import Path
import hashlib
import shutil
import json
import numpy as np
import torch
import torchvision
import onnx
from onnxconverter_common import float16

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/nlf'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

NAMES = ['nose_coco','leye_coco','reye_coco','lear_coco','rear_coco',
         'lsho_coco','rsho_coco','lelb_coco','relb_coco','lwri_coco','rwri_coco',
         'lhip_coco','rhip_coco','lkne_coco','rkne_coco','lank_coco','rank_coco',
         'head_smpl','neck_coco','pelv_coco','lfoo_coco','rfoo_coco',
         'ltoe_coco','rtoe_coco','lhee_coco','rhee_coco']

class FixedPoints(torch.nn.Module):
    def __init__(self, model, w, b):
        super().__init__()
        self.backbone = model.backbone
        self.link = model.heatmap_head.layer
        self.register_buffer('w', w.float().flatten(0,1).unsqueeze(-1).unsqueeze(-1))
        self.register_buffer('b', b.float().flatten())
        self.register_buffer('axis', torch.linspace(0,1,8))
        self.uncert_bias = float(model.heatmap_head.uncert_bias)
        self.uncert_bias2 = float(model.heatmap_head.uncert_bias2)

    def forward(self, image):
        image = image.pow(.8/2.2)
        f = self.link(self.backbone(image))
        logits = torch.nn.functional.conv2d(f, self.w, self.b).float().reshape(1,26,10,8,8)
        metric = logits[:,:,1].flatten(2).softmax(-1).reshape(1,26,8,8)
        heat = logits[:,:,2:].flatten(2).softmax(-1).reshape(1,26,8,8,8)
        uvx = (heat.sum((2,3)) * self.axis).sum(-1)
        uvy = (heat.sum((2,4)) * self.axis).sum(-1)
        z = (heat.sum((3,4)) * self.axis).sum(-1) * 2.2
        x = ((metric.sum(2) * self.axis).sum(-1) * 224 + 16) * (2.2/256)
        y = ((metric.sum(3) * self.axis).sum(-1) * 224 + 16) * (2.2/256)
        uncert = torch.nn.functional.softplus((logits[:,:,0]*heat.sum(2)).sum((2,3)) + self.uncert_bias) + self.uncert_bias2
        return torch.stack((uvx*224+16,uvy*224+16,x,y,z,uncert),-1)

def main():
    m = torch.jit.load(str(OUT/'source/nlf_s_multi_0.2.2.torchscript'),map_location='cpu').float().eval()
    names = m.per_skeleton_joint_names['']
    indices = [names.index(n) for n in NAMES]
    canonical = m.crop_model.canonical_locs()[indices].detach()
    with torch.no_grad():
        weights = m.crop_model.get_weights_for_canonical_points(canonical)
        wrapper = FixedPoints(m.crop_model,weights['w_tensor'],weights['b_tensor']).eval()
        image = torch.rand(1,3,256,256)
        actual = wrapper(image)
        h = m.crop_model.heatmap_head
        features = m.crop_model.get_features(image.pow(.8/2.2))
        reference = h.decode_features_multi_same_weights(features, {k:v.float() for k,v in weights.items()}, torch.tensor([False]))
        expected = torch.cat((reference[0],reference[1],reference[2].unsqueeze(-1)),-1)
        diff = (actual-expected).abs().max().item()
        assert diff < 0.0001, diff
        traced = torch.jit.trace(wrapper,image,check_trace=False)
        traced = torch.jit.freeze(traced)
        traced.save(str(OUT/'reference.pt'))
        print('Fixed-point reference matches upstream:',diff,flush=True)
        torch.onnx.export(traced,(image,),str(OUT/'pose-fp32.onnx'),input_names=['image'],output_names=['points'],
                          opset_version=17,dynamo=False)
    graph = onnx.load(OUT/'pose-fp32.onnx')
    # Legacy TorchScript emits redundant float-to-float casts. Remove them before
    # mixed conversion; otherwise the converter can leave stale Cast attributes.
    for node in graph.graph.node:
        if node.op_type=='Cast' and any(a.name=='to' and a.i==onnx.TensorProto.FLOAT for a in node.attribute):
            node.op_type='Identity';del node.attribute[:]
    blocked = list(set(float16.DEFAULT_OP_BLOCK_LIST+['Softmax','ReduceMean','ReduceSum','Pow','Sqrt','Div','Softplus']))
    graph = float16.convert_float_to_float16(graph,keep_io_types=True,op_block_list=blocked,disable_shape_infer=False,min_positive_val=1e-8)
    onnx.checker.check_model(graph)
    onnx.save(graph,OUT/'pose-fp16.onnx')
    # FP16 is retained for research; difficult-crop precision audit selected TF32/FP32.
    shutil.copyfile(OUT/'pose-fp32.onnx',OUT/'pose.onnx')
    for name in ['pose.onnx','pose-fp32.onnx','pose-fp16.onnx']:
        digest = hashlib.file_digest((OUT/name).open('rb'),'sha256').hexdigest()
        (OUT/(name+'.sha256')).write_text(digest+'\n')
    (OUT/'points.json').write_text(json.dumps(dict(names=NAMES,canonical_points=canonical.tolist(),
        output='1 x 26 x 6: crop_u_px,crop_v_px,x_m,y_m,z_m,raw_uncertainty_m',
        backbone_parameters=sum(p.numel() for p in wrapper.backbone.parameters()),
        upstream_decode_max_difference=diff,fix_uncert_factor=bool(h.fix_uncert_factor),
        preprocessing='Perspective crop, linear-light bilinear sampling, white padding; gamma 0.8 on GPU; input linear-light [0,1].'),indent=2))
    print('Exported', (OUT/'pose.onnx').stat().st_size,'bytes',flush=True)

if __name__ == '__main__':
    main()

