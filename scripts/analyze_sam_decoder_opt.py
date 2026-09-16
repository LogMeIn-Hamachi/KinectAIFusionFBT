"""Compare native decoder candidates with the currently accepted FP8 pipeline."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/sam-decoder-opt'
a=np.genfromtxt(OUT/'baseline.csv',delimiter=',',names=True)
raw_a=np.fromfile(OUT/'baseline.csv.sam3d.bin',np.float32).reshape(301,70,5)
stats=lambda x:dict(median=float(np.median(x)),p95=float(np.quantile(x,.95)),maximum=float(np.max(x)))
report=dict(scope='Same saved 301-frame replay with accepted FP8 encoder; unchanged calibration, geometry, smoothing and cadence. Fidelity to previous implementation, not physical accuracy.',baseline_timings={k:stats(a[k]) for k in ('inference_ms','decoder_ms','encoder_ms')},candidates={})
for name in ('sampling','sampling-bf16'):
 b=np.genfromtxt(OUT/(name+'.csv'),delimiter=',',names=True)
 assert len(a)==len(b)==301 and np.array_equal(a['depth_id'],b['depth_id'])
 raw=np.fromfile(OUT/(name+'.csv.sam3d.bin'),np.float32).reshape(301,70,5)
 assert np.isfinite(raw).all()
 row=dict(raw_landmark_mm=stats(np.linalg.norm(raw[:,:,:3]-raw_a[:,:,:3],axis=-1)*1000),timings={k:stats(b[k]) for k in ('inference_ms','decoder_ms','encoder_ms')},trackers={})
 for tracker in ('hip','left','right'):
  pos=[np.stack([d[tracker+'_'+c] for c in 'xyz'],axis=-1) for d in (a,b)]
  q=[np.stack([d[tracker+'_q'+c] for c in 'wxyz'],axis=-1) for d in (a,b)];q=[x/np.linalg.norm(x,axis=-1,keepdims=True) for x in q]
  distance=np.linalg.norm(pos[0]-pos[1],axis=-1)*1000
  angle=np.rad2deg(2*np.arccos(np.clip(abs(np.sum(q[0]*q[1],axis=-1)),0,1)))
  angle=np.where(np.all(q[0]==q[1],axis=-1),0.,angle)
  row['trackers'][tracker]=dict(position_mm=stats(distance),rotation_degrees=stats(angle),valid_frames=int(b[tracker+'_valid'].sum()),learned_frames=int(b[tracker+'_AI_position'].sum()))
 row['time_reduction_percent']=100*(1-row['timings']['inference_ms']['median']/report['baseline_timings']['inference_ms']['median'])
 report['candidates'][name]=row
(OUT/'comparison.json').write_text(json.dumps(report,indent=2))
for name,row in report['candidates'].items():
 print(name,'pipeline ms',row['timings']['inference_ms']['median'],'decoder ms',row['timings']['decoder_ms']['median'],'max raw mm',row['raw_landmark_mm']['maximum'])
 print('max tracker mm',max(t['position_mm']['maximum'] for t in row['trackers'].values()),'degrees',max(t['rotation_degrees']['maximum'] for t in row['trackers'].values()))
print('baseline',report['baseline_timings'])
