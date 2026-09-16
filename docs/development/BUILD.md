# Development setup

## Clone and models

Install Git and Git LFS, clone the repository, then run `git lfs install` and `git lfs pull`. The repository contains both deployed SAM variants plus the original checkpoint and MHR rig. A normal source ZIP from GitHub may contain LFS pointers; use a Git clone with LFS for development. Keep the included SAM/DINOv3 agreements with these assets.

## Windows toolchain

- Visual Studio 2022 or its Build Tools with Desktop development with C++, MSVC x64 and a Windows SDK.
- CMake 3.24 or later.
- Microsoft Kinect SDK 1.8 and SDK 2.0 for full application compilation. Sensor runtime-only installations are sufficient for running an existing binary but do not provide development headers.
- Python 3.13 for the native decoder's LibTorch package and optional export tools.
- CUDA Toolkit 12.9 for compiling the native decoder. The selected PyTorch runtime is CUDA 12.8; the CUDA development toolkit and runtime wheel versions are deliberately separate, matching the working build.
- A supported NVIDIA driver/GPU for actual model inference. Tested device: RTX 5070 Ti. Core tests do not need one.

From the repository root in PowerShell:

```powershell
./scripts/bootstrap.ps1
py -3.13 -m venv third_party/sam3d_env
./third_party/sam3d_env/Scripts/python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
./scripts/build.ps1
./third_party/sam3d_env/Scripts/python.exe scripts/prepare_runtime.py
```

The last command creates a fresh `release/dev` folder. Open its KinectRGBD.exe. It refuses to overwrite an existing installation. Run its SteamVR installer manually only when you want to register this development copy; do not register two folders containing the same driver at once.

The bootstrap verifies SHA-256 hashes for ONNX Runtime, TensorRT RTX and OpenVR. NuiSensorLib's header/static library and MIT notice are vendored separately. The runtime assembly copies required DLLs from the pinned downloads, Torch wheel, native build and Visual Studio redistributables. Neither setup script installs sensor/display drivers or edits the system PATH.

If the toolchain lives elsewhere, configure these cache variables before building:

```powershell
cmake --preset windows-x64 -DKINECT_SDK="C:/path/to/v1.8" -DKINECT_V2_SDK="C:/path/to/v2.0_1409" -DSAM_CUDA_ROOT="C:/path/to/CUDA/v12.9" -DSAM_TORCH="C:/path/to/torch"
cmake --build --preset release
ctest --preset release
```

`ORT_ROOT` and `OPENVR_ROOT` can also be overridden. An existing working local environment need not be reinstalled.

## Core tests only

```powershell
./scripts/build.ps1 -Bootstrap -CoreOnly
```

This builds all ten automated suites without Kinect hardware, SDK headers, GPU inference, camera data or LFS model downloads. CI uses this route. It does not validate GPU inference or real VRChat behavior.

## Optional model reference/export work

Install `requirements-models.txt` into the above Python environment after installing Torch/torchvision. It records the working optional analysis/export package versions. Then check out upstream sources at the commits recorded in docs/model-provenance and THIRD_PARTY_NOTICES.md:

- Fast SAM: `third_party/sam3d_research/fast`, commit `808b53c7d9c26a7e511d31144f1e5efb058e15c9` from yangtiming/Fast-SAM-3D-Body.
- DINOv3: `third_party/sam3d_research/dinov3`, commit `6876159a11b4df116f30f667f8c9888617df0751` from facebookresearch/dinov3.
- Momentum: `third_party/sam3d_research/momentum`, commit `11172f05996c1009cae6c5ae8e01654d4f2e270b` from facebookincubator/momentum.

The original `model.ckpt`, `model_config.yaml` and `mhr_model.pt` are included at the root for the existing export scripts. The deployed models in assets are sufficient to build/run the app; re-exporting is optional. Historical benchmark/export scripts may expect intermediate artifacts or privately recorded samples. Those data are intentionally not shared and are not needed by the native build or core tests.

## Packaging

First assemble a fresh runtime folder, then run:

```powershell
./third_party/sam3d_env/Scripts/python.exe scripts/package_distribution.py stage --source release/dev --name KinectFBT-Custom-Windows-x64
./third_party/sam3d_env/Scripts/python.exe scripts/package_distribution.py archive --source release/dev --name KinectFBT-Custom-Windows-x64
```

Packaging uses an explicit allowlist, checks model hashes, and verifies every ZIP entry. Builds/releases stay outside Git. See HANDOFF.md for the previously validated baseline and known limitations.
