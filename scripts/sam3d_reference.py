"""Local development reference/export for the pinned Fast SAM 3D Body source.

Never used by the installed tracking application. No detector, FOV estimator,
SMPL conversion, network download or image upload occurs here.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/sam3d_research/fast"
os.environ.update(USE_COMPILE="0", USE_COMPILE_BACKBONE="0", USE_TRT_BACKBONE="0",
                  MOMENTUM_ENABLED="0", MHR_USE_CUDA_GRAPH="0", MHR_NO_CORRECTIVES="0",
                  SKIP_KEYPOINT_PROMPT="1", BODY_INTERM_PRED_LAYERS="0,1,2",
                  KEYPOINT_PROMPT_INTERM_INTERVAL="999", DEBUG_NAN="0",
                  DEBUG_HAND_PREP="0", DEBUG_BACKBONE_INPUT="0", LAYER_DTYPE="fp32",
                  COMPILE_WARMUP_BATCH_SIZES="")
sys.path.insert(0, str(SOURCE))
import numpy as np
import cv2
import torch

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
_hub_load = torch.hub.load


def local_hub(repo, *args, **kwargs):
    if repo != "facebookresearch/dinov3":
        raise RuntimeError("Unexpected architecture source: " + str(repo))
    kwargs.pop("trust_repo", None)
    kwargs["source"] = "local"
    return _hub_load(str(ROOT / "third_party/sam3d_research/dinov3"), *args, **kwargs)


torch.hub.load = local_hub
from sam_3d_body.build_models import load_sam_3d_body
from sam_3d_body.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
from sam_3d_body.data.utils.prepare_batch import prepare_batch
from torchvision.transforms import ToTensor


def load_model():
    model, cfg = load_sam_3d_body(str(ROOT / "model.ckpt"), device="cuda", mhr_path=str(ROOT / "mhr_model.pt"))
    model.float().eval()
    model.backbone_dtype = torch.float32
    # Hand modules are not called by body mode and need not remain GPU resident.
    for name in ("head_pose_hand", "head_camera_hand", "decoder_hand"):
        getattr(model, name).cpu()
    torch.cuda.empty_cache()
    return model, cfg


def sample_batch(sample, cfg):
    info = json.loads(sample.with_suffix(".json").read_text())
    rgb = cv2.cvtColor(cv2.imread(str(sample)), cv2.COLOR_BGR2RGB)
    projection = np.asarray(info["projection"], dtype=np.float64).reshape(3, 4)
    camera, rotation, center, *_ = cv2.decomposeProjectionMatrix(projection)
    camera /= camera[2, 2]
    # SDK RGB coordinates may contain a reflection. Keep positive focal lengths
    # for the model; retain original P for subsequent metric registration.
    camera[0, 0] = abs(camera[0, 0]); camera[1, 1] = abs(camera[1, 1])
    camera[0, 1] = 0  # Upstream ray/projection code assumes zero skew.
    transform = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=cfg.MODEL.IMAGE_SIZE, use_udp=False), VisionTransformWrapper(ToTensor())])
    batch = prepare_batch(rgb, transform, np.array([info["bbox"]], dtype=np.float32), cam_int=torch.tensor(camera[None], dtype=torch.float32))
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    return batch, info


class BackboneGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone.encoder
        self.register_buffer("mean", model.image_mean.detach().clone())
        self.register_buffer("std", model.image_std.detach().clone())

    def forward(self, image):
        return self.backbone.get_intermediate_layers((image-self.mean)/self.std, n=1, reshape=True, norm=True)[-1]


class ReducedCameraEncoder(torch.nn.Module):
    def __init__(self,original):
        super().__init__()
        self.camera=original.camera;self.conv=original.conv;self.norm=original.norm

    def forward(self,features,rays):
        rays=rays.permute(0,2,3,1)
        rays=torch.cat((rays,torch.ones_like(rays[...,:1])),dim=-1)
        encoded=self.camera(pos=rays.reshape(1,-1,3)).reshape(1,32,32,99).permute(0,3,1,2)
        return self.norm(self.conv(torch.cat((features,encoded),dim=1)))


class FixedBodyPrompt(torch.nn.Module):
    """Body-only deployment uses the same fixed no-user-prompt token every frame."""
    def __init__(self,original):
        super().__init__()
        embedding,mask=original(keypoints=torch.tensor([[[0.,0.,-2.]]],device='cuda'))
        self.register_buffer('embedding',embedding.detach());self.register_buffer('mask',mask.detach())
        self.register_buffer('dense',original.get_dense_pe((32,32)).detach())
        self.no_mask_embed=original.no_mask_embed
    def forward(self,keypoints=None):return self.embedding,self.mask
    def get_dense_pe(self,size):return self.dense


class DecoderGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Antialiased resizing is linear. Resize the fixed pixel-coordinate grid
        # once, then apply the changing crop and camera transform to that grid.
        grid=torch.stack(torch.meshgrid(torch.arange(512),torch.arange(512),indexing="xy"),dim=0)[None].float()
        grid=torch.nn.functional.interpolate(grid,scale_factor=1/16,mode="bilinear",align_corners=False,antialias=True)
        self.register_buffer("ray_pixels",grid.cuda())
        model.ray_cond_emb=ReducedCameraEncoder(model.ray_cond_emb)

    def forward(self, features, center, scale, camera, image_wh):
        m = self.model
        image = features.new_zeros(1,1,3,512,512)
        sx, sy = 512/scale[:,0], 512/scale[:,1]
        zero = sx*0
        affine = torch.stack((sx,zero,256-center[:,0]*sx,zero,sy,256-center[:,1]*sy),dim=-1).reshape(1,1,2,3)
        batch = {"img":image,"bbox_center":center[:,None],"bbox_scale":scale[:,None],
                 "cam_int":camera,"ori_img_size":image_wh[:,None],"img_size":features.new_full((1,1,2),512),
                 "affine_trans":affine,"mask":features.new_zeros(1,1,1,512,512),
                 "mask_score":features.new_zeros(1,1),"person_valid":features.new_ones(1,1)}
        m._initialize_batch(batch)
        m.body_batch_idx=[0];m.hand_batch_idx=[]
        rx=(self.ray_pixels[:,0]*scale[:,0,None,None]/512+center[:,0,None,None]-scale[:,0,None,None]/2-camera[:,0,2,None,None])/camera[:,0,0,None,None]
        ry=(self.ray_pixels[:,1]*scale[:,1,None,None]/512+center[:,1,None,None]-scale[:,1,None,None]/2-camera[:,1,2,None,None])/camera[:,1,1,None,None]
        batch["ray_cond"]=torch.stack((rx,ry),dim=1)
        features=features+m.prompt_encoder.no_mask_embed.weight.reshape(1,-1,1,1)
        prompt=features.new_tensor([[[0,0,-2]]])
        _,out=m.forward_decoder(features,keypoints=prompt,condition_info=m._get_decoder_condition(batch),batch=batch)
        if isinstance(out,list):out=out[-1]
        return out["pred_keypoints_3d"]+out["pred_cam_t"][:,None,:],out["pred_keypoints_2d"],out["joint_global_rots"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-backbone", action="store_true")
    parser.add_argument("--export-decoder", action="store_true")
    parser.add_argument("--export-torchscript", action="store_true")
    parser.add_argument("--landmark-rig", action="store_true")
    parser.add_argument("--save-inputs", action="store_true")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    print("torch", torch.__version__, "GPU", torch.cuda.get_device_name(0), flush=True)
    start = time.perf_counter()
    model, cfg = load_model()
    if args.landmark_rig:
        from sam3d_rig import LandmarkRig,validate_rig
        reduced=LandmarkRig(model.head_pose.mhr,model.head_pose.keypoint_mapping).cuda()
        rig_report=validate_rig(model.head_pose.mhr,reduced)
        (args.output/"rig-validation.json").write_text(json.dumps(rig_report,indent=2))
        model.head_pose.mhr=reduced
        model.head_pose.keypoint_mapping=torch.nn.Parameter(reduced.mapping,requires_grad=False)
        torch.cuda.empty_cache()
        print("Landmark rig validated; retained vertices",reduced.vertex_count,flush=True)
    print("Loaded in", time.perf_counter()-start, "seconds", flush=True)
    results = []
    samples = sorted((ROOT / "artifacts/sam3d/replay-samples").glob("*.bmp"), key=lambda p:int(p.stem))[:args.limit]
    with torch.inference_mode():
        for sample in samples:
            batch, info = sample_batch(sample, cfg)
            if args.save_inputs:
                folder=args.output/(sample.stem+"-inputs");folder.mkdir()
                tensors=(batch["img"].reshape(1,3,512,512),batch["bbox_center"].reshape(1,2),
                         batch["bbox_scale"].reshape(1,2),batch["cam_int"])
                for i,t in enumerate(tensors):
                    t.cpu().numpy().tofile(folder/f"input_{i}.bin")
                    (folder/f"input_{i}.shape").write_text(" ".join(map(str,t.shape))+"\n")
            model._initialize_batch(batch)
            torch.cuda.synchronize(); start = time.perf_counter()
            output = model.forward_step(batch, decoder_type="body")["mhr"]
            torch.cuda.synchronize(); elapsed = (time.perf_counter()-start)*1000
            points = output["pred_keypoints_3d"]+output["pred_cam_t"][:,None,:]
            values = {"camera_points":points, "image_points":output["pred_keypoints_2d"],
                      "joint_rotations_mhr":output["joint_global_rots"], "camera_translation":output["pred_cam_t"]}
            arrays = {k:v.cpu().numpy() for k,v in values.items()}
            if not all(np.isfinite(v).all() for v in arrays.values()):
                raise RuntimeError("Nonfinite reference body output")
            np.savez(args.output/(sample.stem+".npz"), **arrays)
            results.append({"frame":int(sample.stem),"ms_including_upstream_debug_sync":elapsed,
                            "output_shapes":{k:list(v.shape) for k,v in arrays.items()}})
            print("REFERENCE", sample.stem, elapsed, flush=True)
        (args.output/"reference.json").write_text(json.dumps({"precision":"FP32, TF32 disabled", "correctives":True,
            "source":"808b53c7d9c26a7e511d31144f1e5efb058e15c9", "samples":results,
            "max_cuda_allocated_bytes":torch.cuda.max_memory_allocated()},indent=2))
        if args.export_backbone:
            wrapper = BackboneGraph(model).eval()
            image = batch["img"].reshape(1,3,512,512)
            inputs = args.output/"backbone-inputs"; inputs.mkdir()
            image.cpu().numpy().tofile(inputs/"input_0.bin")
            (inputs/"input_0.shape").write_text("1 3 512 512\n")
            reference = wrapper(image).cpu().numpy()
            reference.tofile(args.output/"backbone-reference.bin")
            print("Exporting backbone",flush=True)
            torch.onnx.export(wrapper, (image,), str(args.output/"backbone.onnx"), input_names=["image"],
                              output_names=["features"], opset_version=18, dynamo=True, external_data=True)
        if args.export_decoder or args.export_torchscript:
            from sam3d_export_ops import install
            install()
            features=BackboneGraph(model)(batch["img"].reshape(1,3,512,512))
            wrapper=DecoderGraph(model).eval()
            args_t=(features,batch["bbox_center"].reshape(1,2),batch["bbox_scale"].reshape(1,2),batch["cam_int"],batch["ori_img_size"].reshape(1,2))
            expected=wrapper(*args_t)
            for i,(actual,wanted) in enumerate(zip(expected,(points,output["pred_keypoints_2d"],output["joint_global_rots"]))):
                # Rebuilding affine coordinates in FP32 changes subpixel rounding.
                # Allow 0.001 image pixels, while retaining 10-micron 3D tolerance.
                torch.testing.assert_close(actual,wanted,atol=1e-3 if i==1 else 1e-5,rtol=1e-5)
            inputs=args.output/"decoder-inputs";inputs.mkdir()
            for i,t in enumerate(args_t):
                t.cpu().numpy().tofile(inputs/f"input_{i}.bin")
                (inputs/f"input_{i}.shape").write_text(" ".join(map(str,t.shape))+"\n")
            np.savez(args.output/"decoder-reference.npz",camera_points=expected[0].cpu().numpy(),image_points=expected[1].cpu().numpy(),joint_rotations_mhr=expected[2].cpu().numpy())
            if args.export_torchscript:
                print("Tracing native TorchScript decoder",flush=True)
                model.prompt_encoder=FixedBodyPrompt(model.prompt_encoder)
                # Lightning's trainer property throws during TorchScript's module
                # inspection. Retain the inference methods/state in a plain
                # nn.Module; training-only Lightning properties are not runtime.
                methods={}
                for cls in reversed(type(model).__mro__):
                    if cls.__module__.startswith('sam_3d_body'):
                        methods.update({k:v for k,v in cls.__dict__.items() if not k.startswith('__')})
                model.__class__=type('NativeSAMBody',(torch.nn.Module,),methods)
                traced=torch.jit.freeze(torch.jit.trace(wrapper,args_t,check_trace=False).eval())
                torch.jit.save(traced,str(args.output/"decoder.pt"))
                traced=torch.jit.load(str(args.output/"decoder.pt")).eval()
                from sam3d_torch_constants import fold
                folded=fold(traced)
                torch.jit.save(traced,str(args.output/"decoder.pt"))
                print("CUDA constant copies folded",folded,flush=True)
                for _ in range(5):actual=traced(*args_t)
                for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-3,rtol=1e-5)
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):traced(*args_t)
                torch.cuda.current_stream().wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):captured=traced(*args_t)
                timings=[]
                for _ in range(30):
                    torch.cuda.synchronize();start=time.perf_counter();graph.replay();torch.cuda.synchronize()
                    timings.append((time.perf_counter()-start)*1000)
                for a,b in zip(captured,expected):torch.testing.assert_close(a,b,atol=1e-3,rtol=1e-5)
                (args.output/"cuda-graph-timing.json").write_text(json.dumps({'median_ms':float(np.median(timings)),'p95_ms':float(np.quantile(timings,.95)),'timings_ms':timings,'scope':'GPU decoder only, inputs and outputs resident on device; Python development check of serialized native module'},indent=2))
                print("CUDA graph decoder median",np.median(timings),flush=True)
                return
            print("Exporting body decoder and MHR rig",flush=True)
            from torch.onnx import symbolic_helper
            def matrix_transpose(g,value):
                rank=symbolic_helper._get_tensor_rank(value)
                if rank is None or rank<2:raise RuntimeError("Unknown rank for matrix transpose")
                order=list(range(rank));order[-1],order[-2]=order[-2],order[-1]
                return g.op("Transpose",value,perm_i=order)
            torch.onnx.register_custom_op_symbolic("aten::mT",matrix_transpose,18)
            wrapper.cpu()
            args_t=tuple(t.cpu() for t in args_t)
            torch.onnx.export(wrapper,args_t,str(args.output/"decoder.onnx"),input_names=["features","center","scale","camera","image_wh"],
                              output_names=["camera_points","image_points","joint_rotations_mhr"],opset_version=18,dynamo=False)
    (args.output/"reference.json").write_text(json.dumps({"precision":"FP32, TF32 disabled", "correctives":True,
        "source":"808b53c7d9c26a7e511d31144f1e5efb058e15c9", "samples":results,
        "max_cuda_allocated_bytes":torch.cuda.max_memory_allocated()},indent=2))


if __name__ == "__main__":
    main()
