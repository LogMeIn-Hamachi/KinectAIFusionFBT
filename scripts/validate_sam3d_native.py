"""Verify native deployment plumbing with synthetic data, NOT SAM model quality."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/dev_python"))
import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from sam3d_assets import seal, compare

EXE = ROOT / "build/Release/kf_sam3d_bench.exe"
BASE = ROOT / "artifacts/sam3d"
BASE.mkdir(exist_ok=True)
folder = Path(tempfile.mkdtemp(prefix="native-check-", dir=BASE))
rng = np.random.default_rng(304)
image = rng.integers(0, 256, (480, 640, 4), dtype=np.uint8)
image.tofile(folder / "test.bgra")
crop_results = []
for i, box in enumerate([(91, 27, 571, 472), (-55, 5, 405, 555), (240, 100, 420, 470), (100, 100, 300, 300)]):
    x1, y1, x2, y2 = box
    scale = np.array([x2-x1, y2-y1], dtype=np.float64) * 1.25
    # Upstream first expands to .75 aspect ratio, then expands to square.
    side = max(scale[1], scale[0] / .75)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    affine = np.array([[512/side, 0, 256-cx*512/side], [0, 512/side, 256-cy*512/side]])
    expected = cv2.warpAffine(image[:, :, :3], affine, (512, 512), flags=cv2.INTER_LINEAR)
    expected = np.ascontiguousarray(expected[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)/255
    output = folder / f"crop_{i}.bin"
    subprocess.run([str(EXE), "crop", str(folder/"test.bgra"), str(output), *map(str, box)], check=True)
    actual = np.fromfile(output, dtype=np.float32).reshape(expected.shape)
    maximum = float(np.abs(expected-actual).max())
    crop_results.append({"bbox": box, "max_error": maximum})
    assert maximum < 1e-6, crop_results[-1]

weights = rng.normal(0, .05, (128, 128)).astype(np.float32)
sample = rng.normal(0, .05, (1, 128)).astype(np.float32)
for precision in ("fp32", "fp16"):
    target = folder / precision
    target.mkdir()
    dtype = np.float32 if precision == "fp32" else np.float16
    kind = TensorProto.FLOAT if precision == "fp32" else TensorProto.FLOAT16
    # A real FP16 MatMul exercises typed GPU inference and FP32 result decoding.
    graph = helper.make_graph([
        helper.make_node("MatMul", ["image_features", "weights"], ["product"]),
        helper.make_node("Cast", ["product"], ["body_features"], to=TensorProto.FLOAT),
    ], "synthetic_runtime_test_not_sam", [helper.make_tensor_value_info("image_features", kind, [1, 128])],
       [helper.make_tensor_value_info("body_features", TensorProto.FLOAT, [1, 128])],
       [numpy_helper.from_array(weights.astype(dtype), "weights")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    path = target / "test.onnx"
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location="test.weights", size_threshold=0)
    seal(path)
    sample.astype(dtype).tofile(target / "input_0.bin")
    (target / "input_0.shape").write_text("1 128\n")
    for backend in (["cpu", "gpu"] if precision == "fp32" else ["gpu"]):
        subprocess.run([str(EXE), "run", str(path), str(EXE.parent), str(target), str(target/backend), backend, "5"], check=True)
    if precision == "fp32":
        compare(target/"cpu", target/"gpu", .0001)
        actual = np.fromfile(target/"cpu/output_0.bin", np.float32)
        assert np.max(np.abs(actual-(sample @ weights).reshape(-1))) < 1e-6
    else:
        compare(folder/"fp32/cpu", target/"gpu", .0002)

# Reject changed weights and malformed input shapes before running the model.
weight_file = folder/"fp16/test.weights"
contents = bytearray(weight_file.read_bytes()); contents[0] ^= 1
weight_file.write_bytes(contents)
bad = subprocess.run([str(EXE), "run", str(folder/"fp16/test.onnx"), str(EXE.parent), str(folder/"fp16"), str(folder/"corrupt"), "gpu", "1"], capture_output=True, text=True)
assert bad.returncode != 0 and "checksum mismatch" in bad.stderr, bad.stderr
assert not (folder/"corrupt").exists()
result = {"scope": "Synthetic deployment tests; no SAM weights or Kinect tracking quality measured", "crops": crop_results,
          "fp32_cpu_gpu": "passed", "mixed_precision_gpu": "passed", "corrupt_external_weights_rejected": True, "artifacts": str(folder)}
(BASE/"native-validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
