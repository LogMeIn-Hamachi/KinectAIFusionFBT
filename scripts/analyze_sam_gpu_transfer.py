"""Compare four sequential native replay runs, including full numeric parity."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/sam-gpu-transfer'
names=['before','after','after-repeat','before-repeat']
runs={name:list(csv.DictReader((OUT/(name+'.csv')).open())) for name in names}
reference=runs['before']
raw=(OUT/'before.csv.sam3d.bin').read_bytes()
report=dict(scope='Sequential A/B/B/A native replay of the same 301 recorded frames; RTX 5070 Ti; no VRChat rendering load. Model, precision, crop, calibration and estimator unchanged.',runs={})
for name,rows in runs.items():
    assert len(rows)==len(reference)==301
    difference=max(abs(float(a[k])-float(b[k])) for a,b in zip(reference,rows) for k in a if not k.endswith('_ms'))
    exact_raw=(OUT/(name+'.csv.sam3d.bin')).read_bytes()==raw
    assert difference==0 and exact_raw
    assert all(all(int(r[k])==1 for k in ('hip_valid','left_valid','right_valid','hip_AI_position','left_AI_position','right_AI_position')) for r in rows)
    times={k:dict(median_ms=float(np.median([float(r[k]) for r in rows])),p95_ms=float(np.percentile([float(r[k]) for r in rows],95))) for k in ('inference_ms','encoder_ms','decoder_ms','crop_ms')}
    report['runs'][name]=dict(frames=len(rows),max_non_timing_difference=difference,raw_prediction_bytes_identical=exact_raw,times=times)
for mode in ('before','after'):
    values=[float(row['inference_ms']) for name in (mode,mode+'-repeat') for row in runs[name]]
    report[mode]=dict(pooled_median_ms=float(np.median(values)),pooled_p95_ms=float(np.percentile(values,95)))
report['median_change_percent']=100*(report['after']['pooled_median_ms']/report['before']['pooled_median_ms']-1)
report['feature_host_transfer_bytes_removed_per_frame']=2*1280*32*32*4
(OUT/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
