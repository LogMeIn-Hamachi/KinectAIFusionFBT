"""Development only. Compare native preprocessing/inference with OpenCV and ORT.

Input is an upstream COCO test photo, never claimed to be Kinect footage.
No Python is included in, or needed by, the portable application.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/dev_python"))
import cv2
import numpy as np
import onnxruntime as ort

ART = ROOT / "artifacts"
EXE = ROOT / "build/Release/kf_validate.exe"
MODEL = ROOT / "assets/pose.onnx"
image = cv2.imread(str(ART / "reference-person.jpg"))
if image is None:
    raise RuntimeError("Download the upstream reference image first")
image = cv2.resize(image, (640, 480))
bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
bgra.tofile(ART / "reference.bgra")

results = {"image": "upstream mmpose tests/data/coco/000000000785.jpg; resized, not Kinect",
           "reference_ort": ort.__version__, "preprocessing": []}
for bbox in [(0, 0, 640, 480), (91, 27, 571, 472), (-55, 5, 405, 555)]:
    x1, y1, x2, y2 = bbox
    w, h = max(20, x2-x1)*1.25, max(20, y2-y1)*1.25
    if w/h > .75:
        h = w/.75
    else:
        w = h*.75
    cx, cy = (x1+x2)/2, (y1+y2)/2
    transform = np.array([[192/w, 0, 96-cx*192/w], [0, 256/h, 128-cy*256/h]], np.float64)
    warped = cv2.warpAffine(image, transform, (192, 256), flags=cv2.INTER_LINEAR)
    reference = ((warped[..., ::-1].astype(np.float32)-[123.675,116.28,103.53])/[58.395,57.12,57.375]).transpose(2,0,1).astype(np.float32)
    subprocess.run([str(EXE), "preprocess", str(ART/"reference.bgra"), str(ART/"native.tensor"), *map(str,bbox)], check=True, cwd=ART)
    native = np.fromfile(ART/"native.tensor", np.float32).reshape(3,256,192)
    error = np.abs(reference-native)
    results["preprocessing"].append({"bbox":bbox,"max_normalized_error":float(error.max()),"mean_normalized_error":float(error.mean()),"fraction_exact":float(np.mean(error<1e-6))})
    assert error.max() < 1e-6, results["preprocessing"][-1]

reference.tofile(ART/"reference.tensor")
session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
expected = session.run(None, {session.get_inputs()[0].name: reference[None]})
expected_flat = np.concatenate([a.flatten() for a in expected])
for backend in ["cpu", "gpu"]:
    completed = subprocess.run([str(EXE), "model", str(MODEL), str(EXE.parent), backend, str(ART/"reference.tensor")], check=True, cwd=ART, capture_output=True, text=True)
    actual = np.fromfile(ART/f"{backend}-outputs.bin", np.float32)
    error = np.abs(expected_flat-actual)
    peak_errors = []
    for ref, pred in [(expected[0].reshape(26,384),actual[:26*384].reshape(26,384)), (expected[1].reshape(26,512),actual[26*384:].reshape(26,512))]:
        peak_errors.extend((np.abs(ref.argmax(1)-pred.argmax(1))/2).tolist())
    results[backend] = {"max_output_error":float(error.max()),"mean_output_error":float(error.mean()),"max_keypoint_axis_difference_input_pixels":max(peak_errors),"benchmark":completed.stdout}
    assert error.max() < (.0002 if backend=="cpu" else .03), results[backend]
    assert max(peak_errors) <= (0 if backend=="cpu" else 1), results[backend]

(ART/"reference-validation.json").write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
