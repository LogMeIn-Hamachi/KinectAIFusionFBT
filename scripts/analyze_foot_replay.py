"""Summarize replay evidence; availability is not ground-truth accuracy."""
import csv, json, sys
from pathlib import Path
import numpy as np

path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open()))
def values(key):
    return np.array([float(row[key]) for row in rows])
report = {"frames": len(rows), "span_s": float(values("host")[-1]-values("host")[0]), "feet": {}}
for side, ankle, toe, heel in [("left",15,20,24),("right",16,21,25)]:
    point = lambda j: np.array([values(f"joint_{j}_{axis}") for axis in "xyz"]).T
    entry = {"accepted_direction_frames":int(np.sum(values(f"{side}_sigma_y_rad") < 1)),
             "heel_toe_m_p10_p50_p90":np.percentile(np.linalg.norm(point(toe)-point(heel),axis=1),[10,50,90]).tolist(),
             "joints":{}}
    for label, joint in [("ankle",ankle),("toe",toe),("heel",heel)]:
        item = {"rgb_score_median":float(np.median(values(f"joint_{joint}_rgb_score"))),
                "depth_source_frames":int(np.sum(values(f"joint_{joint}_source")==2))}
        if f"joint_{joint}_depth_rejection" in rows[0]:
            causes, counts = np.unique(values(f"joint_{joint}_depth_rejection"),return_counts=True)
            item["rejections"] = {str(int(k)):int(v) for k,v in zip(causes,counts)}
            item["player_samples_median"] = float(np.median(values(f"joint_{joint}_player_samples")))
            item["depth_spread_p50_p90"] = np.percentile(values(f"joint_{joint}_depth_spread"),[50,90]).tolist()
        entry["joints"][label] = item
    report["feet"][side] = entry
path.with_suffix('.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
