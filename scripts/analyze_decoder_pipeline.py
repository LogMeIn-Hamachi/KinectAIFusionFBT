"""Summarize matched native FP32/TF32 replays; this is not a ground-truth test."""
from pathlib import Path
import json
import numpy as np

root=Path(__file__).resolve().parents[1]
base=root/'artifacts/sam3d'
data={m:np.genfromtxt(base/f'decoder-pipeline-{m}.csv',delimiter=',',names=True) for m in ('fp32','tf32')}
a,b=data.values()
assert len(a)==len(b)==301 and np.array_equal(a['depth_id'],b['depth_id']) and np.array_equal(a['host'],b['host'])
raw={m:np.fromfile(base/f'decoder-pipeline-{m}.csv.sam3d.bin',np.float32).reshape(301,70,5) for m in data}
assert all(np.isfinite(x).all() for x in raw.values())
delta=np.linalg.norm(raw['fp32'][:,:,:3]-raw['tf32'][:,:,:3],axis=-1)*1000
report={'scope':'301 matching recorded frames, offline native pipeline, unchanged settling settings; no live VRChat GPU load and no new recording. Numerical comparison, not tracking accuracy.',
        'raw_landmark_difference_mm':{'mean':float(delta.mean()),'p95':float(np.quantile(delta,.95)),'max':float(delta.max())},'timings':{},'trackers':{}}
for mode,d in data.items():
    report['timings'][mode]={key:{'median_ms':float(np.median(d[key][10:])), 'p95_ms':float(np.quantile(d[key][10:],.95))} for key in ('inference_ms','encoder_ms','decoder_ms')}

def angular(q,r):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);r=r/np.linalg.norm(r,axis=-1,keepdims=True)
    return np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(q*r,axis=-1)),0,1)))

def variation(d,name,mask):
    pos=np.stack([d[name+'_'+c][mask] for c in 'xyz'],axis=-1)
    q=np.stack([d[name+'_q'+c][mask] for c in 'wxyz'],axis=-1)
    return {'position_half_second_difference_rms_mm':float(np.sqrt(np.mean(np.sum((np.diff(pos,n=2,axis=0)/2)**2,axis=-1)))*1000),
            'rotation_step_rms_deg':float(np.sqrt(np.mean(angular(q[1:],q[:-1])**2)))}

t=a['host']-a['host'][0]
for name in ('hip','left','right'):
    p=[np.stack([d[name+'_'+c] for c in 'xyz'],axis=-1) for d in (a,b)]
    q=[np.stack([d[name+'_q'+c] for c in 'wxyz'],axis=-1) for d in (a,b)]
    diff=np.linalg.norm(p[0]-p[1],axis=-1)*1000
    angle=angular(*q)
    row={'position_difference_mm':{'median':float(np.median(diff)),'p95':float(np.quantile(diff,.95)),'max':float(diff.max())},
         'rotation_difference_deg':{'median':float(np.median(angle)),'p95':float(np.quantile(angle,.95)),'max':float(angle.max())},
         'valid_frames':{m:int(d[name+'_valid'].sum()) for m,d in data.items()},
         'SAM_position_frames':{m:int(d[name+'_SAM_position'].sum()) for m,d in data.items()},'intervals':[]}
    for i in range(10):
        mask=(t>=i)&(t<i+1)
        row['intervals'].append({'start_s':i,**{m:variation(d,name,mask) for m,d in data.items()}})
    report['trackers'][name]=row

profiles={}
for mode in data:
    kernels=json.loads((base/f'decoder-kernels-{mode}/kernels.json').read_text())
    tensor={n:v for n,v in kernels.items() if 'tensorop' in n}
    profiles[mode]={'scope':'Development profiler; identical serialized model, CUDA graph and precision flags. Three graph replays; kernel identities, not hardware utilization counters.', 'tensorop_kernel_launches':sum(x['count'] for x in tensor.values()),'tensorop_kernels':tensor}
report['kernel_trace']=profiles
(base/'decoder-pipeline-comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='trackers'},indent=2))
for name,r in report['trackers'].items():print(name,json.dumps({k:v for k,v in r.items() if k!='intervals'}))
