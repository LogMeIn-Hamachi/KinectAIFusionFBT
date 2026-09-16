"""Numerical and timing gates for the native selective-FP8 replay."""
from pathlib import Path
import json
import argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/sam-fp8'
parser=argparse.ArgumentParser();parser.add_argument('candidate',nargs='?',default='fp8');args=parser.parse_args()
paths={'baseline':'baseline','fp8':args.candidate}
data={name:np.genfromtxt(OUT/f'{path}.csv',delimiter=',',names=True) for name,path in paths.items()}
a,b=data.values()
assert len(a)==len(b)==301 and np.array_equal(a['depth_id'],b['depth_id'])
raw={name:np.fromfile(OUT/f'{path}.csv.sam3d.bin',np.float32).reshape(301,70,5) for name,path in paths.items()}
assert all(np.isfinite(v).all() for v in raw.values())
delta=np.linalg.norm(raw['baseline'][:,:,:3]-raw['fp8'][:,:,:3],axis=-1)*1000
stats=lambda x:dict(median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(np.max(x)))
report=dict(scope='Same 301 saved frames, native GPU-resident pipeline. Six calibration frames overlap; 295 frames not used to fit scales. Numerical fidelity, not physical accuracy or live VRChat FPS.',raw_landmarks_mm=stats(delta),timings={},trackers={})
for name,d in data.items():report['timings'][name]={key:stats(d[key]) for key in ('inference_ms','encoder_ms','decoder_ms')}
accepted=True
for name in ('hip','left','right'):
    positions=[np.stack([d[name+'_'+c] for c in 'xyz'],axis=-1) for d in (a,b)]
    quats=[np.stack([d[name+'_q'+c] for c in 'wxyz'],axis=-1) for d in (a,b)]
    quats=[q/np.linalg.norm(q,axis=-1,keepdims=True) for q in quats]
    pos=np.linalg.norm(positions[0]-positions[1],axis=-1)*1000
    angle=np.rad2deg(2*np.arccos(np.clip(abs(np.sum(quats[0]*quats[1],axis=-1)),0,1)))
    valid={m:int(d[name+'_valid'].sum()) for m,d in data.items()}
    source={m:int(d[name+'_AI_position'].sum()) for m,d in data.items()}
    report['trackers'][name]=dict(position_mm=stats(pos),rotation_deg=stats(angle),valid=valid,learned=source)
    accepted &= max(pos)<10 and np.quantile(pos,.95)<5 and max(angle)<2 and np.quantile(angle,.95)<1 and valid['fp8']==valid['baseline']==301 and source['fp8']==source['baseline']==301
report['fidelity_gate_passed']=bool(accepted)
report['fidelity_gate']='Each tracker: position p95 <5 mm / max <10 mm, angle p95 <1 degree / max <2 degrees, no lost valid or learned frames.'
report['median_speedup_percent']=100*(1-report['timings']['fp8']['inference_ms']['median']/report['timings']['baseline']['inference_ms']['median'])
(OUT/(args.candidate+'-pipeline-comparison.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
