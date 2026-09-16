"""Compare native preprocessing/reconstruction and NVIDIA inference to official NLF.

Only uses previously authorized local samples. Numeric agreement is not tracking
ground truth. GPU latency includes upload/readback and native image preprocessing.
"""
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
import torch
import torchvision
from nlf_export import NAMES

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nlf-validation'
OUT.mkdir(exist_ok=True)
EXE=ROOT/'build/Release/kf_nlf_bench.exe'
parser=argparse.ArgumentParser();parser.add_argument('--gpu',action='store_true');parser.add_argument('--precision',default='pose.onnx');args=parser.parse_args()
torch.set_num_threads(4)
m=torch.jit.load(str(ROOT/'assets/nlf/source/nlf_s_multi_0.2.2.torchscript'),map_location='cpu').float().eval()
model=torch.jit.load(str(ROOT/'assets/nlf/reference.pt'),map_location='cpu').eval()
report=[]

def run(argv,log):
    with log.open('w') as f:
        subprocess.run(list(map(str,argv)),stdout=f,stderr=subprocess.STDOUT,check=True)

with torch.no_grad():
    samples=sorted((ROOT/'artifacts/sam3d/replay-samples').glob('*.json'),key=lambda p:int(p.stem))
    for i,info_path in enumerate(samples):
        info=json.loads(info_path.read_text())
        im=cv2.imread(str(info_path.with_suffix('.bmp')),cv2.IMREAD_UNCHANGED)
        if im.shape[2]==3: im=cv2.cvtColor(im,cv2.COLOR_BGR2BGRA)
        P=np.array(info['projection']).reshape(3,4)
        a,b,c=P[0,:3],P[1,:3],P[2,:3]
        scale=np.linalg.norm(c);z=c/scale;cy=np.dot(b/scale,z);y=b/scale-z*cy;fy=np.linalg.norm(y);y/=fy
        cx=np.dot(a/scale,z);skew=np.dot(a/scale,y);x=a/scale-z*cx-y*skew;fx=np.linalg.norm(x)
        K=torch.tensor([[[fx,0,cx],[0,fy,cy],[0,0,1]]],dtype=torch.float32)
        x1,y1,x2,y2=info['bbox'];w,h=(x2-x1)*1.25,(y2-y1)*1.25
        center=((x1+x2)/2,(y1+y2)/2)
        # Extra off-centre/padded boxes exercise camera rotation and pyramid levels.
        if i==1:center=(-20,200);w,h=300,600
        if i==2:center=(610,320);w,h=250,310
        prefix=OUT/str(i)
        im.tofile(str(prefix)+'.bgra')
        Path(str(prefix)+'.txt').write_text(' '.join(map(str,K.numpy().flatten().tolist()+[*center,w,h])))
        boxes=torch.tensor([[center[0]-w/2,center[1]-h/2,w,h]],dtype=torch.float32)
        images=torch.from_numpy(im[:,:,:3][:,:,::-1].copy()).permute(2,0,1)[None].float()/255
        crop,newK,R=m._get_crops(images.pow(2.2),K,torch.zeros(1,5),torch.tensor([[0.,-1.,0.]]),boxes,
            torch.tensor([0]),torch.eye(3)[None],torch.ones(1),torch.tensor([.8]),1)
        crop=crop[0].pow(2.2/.8)
        raw=model(crop)
        raw.numpy().tofile(str(prefix)+'.points.bin')
        decoded,unc=m.crop_model.heatmap_head.reconstruct_absolute(raw[:,:,:2],raw[:,:,2:5],raw[:,:,5],newK[0])
        expected=(decoded/1000) @ R[0,0]
        run([EXE,'geometry',prefix],OUT/f'{i}-geometry.log')
        native_crop=np.fromfile(str(prefix)+'.crop.bin',np.float32).reshape(1,3,256,256)
        native=np.genfromtxt(str(prefix)+'.geometry.csv',delimiter=',',names=True)
        xyz=np.column_stack([native[a] for a in 'xyz'])
        crop_error=float(np.max(np.abs(native_crop-crop.numpy())))
        xyz_error=float(np.max(np.linalg.norm(xyz-expected.numpy()[0],axis=1)))*1000
        assert crop_error<.001, (i,crop_error)
        assert xyz_error<.1, (i,xyz_error)
        row=dict(sample=i,crop_max_error=crop_error,reconstruction_max_mm=xyz_error)
        if args.gpu:
            # Reference runs on identical native pixels, separating geometry and precision error.
            raw_native=model(torch.from_numpy(native_crop))
            decoded,_=m.crop_model.heatmap_head.reconstruct_absolute(raw_native[:,:,:2],raw_native[:,:,2:5],raw_native[:,:,5],newK[0])
            reference=(decoded/1000)@R[0,0]
            reference.numpy().tofile(str(prefix)+'.reference3d.bin')
            run([EXE,'model',prefix,ROOT/'release/KinectSAM3D-Preview',ROOT/'assets/nlf'/args.precision,'30'],OUT/f'{i}-gpu.log')
            gpu=np.genfromtxt(str(prefix)+'.model.csv',delimiter=',',names=True)
            error=np.linalg.norm(np.column_stack([gpu[a] for a in 'xyz'])-reference.numpy()[0],axis=1)*1000
            row.update(nvidia_max_mm=float(error.max()),nvidia_mean_mm=float(error.mean()),timings=(OUT/f'{i}-gpu.log').read_text())
            row['precision_limit_mm']=10 if i in (1,2) else 1
            assert error.max()<row['precision_limit_mm'],row
        report.append(row)
        print(json.dumps(row),flush=True)
(OUT/('validation-'+args.precision+'.json')).write_text(json.dumps(report,indent=2))
