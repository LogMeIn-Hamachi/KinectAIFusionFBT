"""Development-only model access, integrity and numeric comparison tools.

No credentials are printed or stored in this repository. The deployed tracker
does not import Python. Download only through Meta's official access gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/dev_python"))
REPO = "facebook/sam-3d-body-dinov3"
FILES = ("model.ckpt", "model_config.yaml", "assets/mhr_model.pt")


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def credential():
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    base = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    path = base / "token"
    return path.read_text().strip() if path.is_file() else None


def request(url):
    token = credential()
    headers = {"Authorization": "Bearer " + token} if token else {}
    # urllib removes Authorization on neither every kind of redirect nor every
    # host. HF large files redirect to signed CDN URLs: drop auth off-origin.
    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            result = super().redirect_request(req, fp, code, msg, headers, newurl)
            from urllib.parse import urlparse
            if result and urlparse(newurl).netloc != "huggingface.co":
                result.remove_header("Authorization")
            return result
    return urllib.request.build_opener(SafeRedirect()).open(
        urllib.request.Request(url, headers=headers), timeout=60)


def access(fetch):
    try:
        with request(f"https://huggingface.co/{REPO}/resolve/main/model_config.yaml") as r:
            r.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            print("Model unavailable: Meta approval and an authorized local Hugging Face login are required.")
            print("No checkpoint downloaded. Never paste your token into chat.")
            return 2
        raise
    print("Official model access verified.")
    if not fetch:
        return 0
    with request(f"https://huggingface.co/api/models/{REPO}?blobs=true") as r:
        metadata = json.load(r)
    revision = metadata["sha"]
    folder = ROOT / "assets/sam3d/checkpoint" / revision
    folder.mkdir(parents=True, exist_ok=True)
    siblings = {s["rfilename"]: s for s in metadata["siblings"]}
    records = []
    for name in FILES:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = siblings[name].get("lfs", {}).get("sha256")
        if path.exists():
            if expected and digest(path) != expected:
                raise RuntimeError(f"Existing checkpoint hash mismatch: {name}")
        else:
            part = path.with_name(path.name + ".partial")
            # A failed transfer is safe to retry; final files are never partial.
            with request(f"https://huggingface.co/{REPO}/resolve/{revision}/{name}") as src, part.open("wb") as dst:
                while chunk := src.read(8 << 20):
                    dst.write(chunk)
            if expected and digest(part) != expected:
                raise RuntimeError(f"Downloaded checkpoint hash mismatch: {name}")
            part.replace(path)
        records.append({"file": name, "sha256": digest(path), "bytes": path.stat().st_size})
        print(f"Verified {name}")
    (folder / "provenance.json").write_text(json.dumps({"repository": REPO, "revision": revision, "files": records}, indent=2))
    print(f"Checkpoint ready in {folder}")
    return 0


def seal(model):
    import onnx
    from onnx.external_data_helper import _get_all_tensors
    model = Path(model).resolve()
    graph = onnx.load(model, load_external_data=False)
    files = {model}
    for tensor in _get_all_tensors(graph):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            fields = {f.key: f.value for f in tensor.external_data}
            location = fields["location"]
            relative = Path(location)
            if relative.is_absolute() or relative.drive or ".." in relative.parts:
                raise ValueError("External weights must stay inside the model directory")
            path = (model.parent / relative).resolve()
            if not path.is_relative_to(model.parent):
                raise ValueError("External weights resolve outside model directory")
            files.add(path)
    onnx.checker.check_model(str(model))
    manifest = Path(str(model) + ".files.sha256")
    lines = [f"{digest(p)}  {p.relative_to(model.parent).as_posix()}" for p in sorted(files)]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Sealed graph and {len(files)-1} external weight files: {manifest}")


def compare(reference, candidate, tolerance):
    import numpy as np
    reference, candidate = Path(reference), Path(candidate)
    a = json.loads((reference / "result.json").read_text())
    b = json.loads((candidate / "result.json").read_text())
    if len(a["outputs"]) != len(b["outputs"]):
        raise ValueError("Output count differs")
    report = []
    types = {1: np.float32, 10: np.float16, 7: np.int64, 6: np.int32, 9: np.bool_}
    for i, (x, y) in enumerate(zip(a["outputs"], b["outputs"])):
        if x["name"] != y["name"] or x["shape"] != y["shape"]:
            raise ValueError("Output names/shapes differ")
        def read(folder, info):
            data = np.fromfile(folder / f"output_{i}.bin", dtype=types[info["onnx_type"]])
            return data.reshape(info["shape"]).astype(np.float64)
        av, bv = read(reference, x), read(candidate, y)
        if not np.isfinite(av).all() or not np.isfinite(bv).all():
            raise ValueError("Nonfinite output")
        error = np.abs(av - bv)
        report.append({"name": x["name"], "mean_absolute": float(error.mean()),
                       "p95_absolute": float(np.quantile(error, .95)), "max_absolute": float(error.max())})
    print(json.dumps(report, indent=2))
    if any(r["max_absolute"] > tolerance for r in report):
        raise ValueError("Numeric tolerance exceeded; this is not a tracking quality benchmark")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("fetch")
    sealing = sub.add_parser("seal")
    sealing.add_argument("model")
    comparing = sub.add_parser("compare")
    comparing.add_argument("reference")
    comparing.add_argument("candidate")
    comparing.add_argument("--tolerance", type=float, required=True)
    args = parser.parse_args()
    if args.command in ("status", "fetch"):
        raise SystemExit(access(args.command == "fetch"))
    if args.command == "seal":
        seal(args.model)
    else:
        compare(args.reference, args.candidate, args.tolerance)
