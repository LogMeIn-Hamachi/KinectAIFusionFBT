# Kinect RGB-D full-body tracking

Native Windows C++20 Kinect v1/v2 tracking for SteamVR and VRChat, using Fast SAM 3D Body plus metric depth. NVIDIA acceleration is required for SAM inference. The current accepted tracking baseline is **Preview 23**.

## Start here

- Developers and AI agents: read [AGENTS.md](AGENTS.md), [build instructions](docs/development/BUILD.md) and [current handoff](docs/development/HANDOFF.md).
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md).
- Application setup: [runtime guide](packaging/windows/README.md).
- Component agreements: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [license texts](docs/licenses).

Clone with Git LFS (`git lfs install`, then `git lfs pull`). Large weights are stored in LFS; a plain GitHub source ZIP may contain pointers rather than model data. The repository includes optimized and original SAM deployment assets, the original checkpoint and MHR rig. Native runtimes are assembled from pinned dependencies, not copied from a private installation.

```powershell
# Core tests only: no camera, GPU, models or Kinect SDK installation needed.
./scripts/build.ps1 -Bootstrap -CoreOnly
```

Full application compilation also needs both Kinect SDKs, MSVC, a matching LibTorch environment and CUDA toolkit; see the build guide. Python is used for development/packaging, not by the running application.

## Current behavior

- SAM optimized: selective FP8 encoder, native CUDA TF32 decoder, shared GPU features and optimized sampling.
- Camera-led body placement and depth-supported foot contact; no headset-gaze-driven foot movement.
- One self-paced, controller-only alignment routine with learned offsets and a separate check pose.
- Native SteamVR waist and feet; optional OSC. OVR playspace movement uses a fixed physical reference.
- Kinect v2 full-HD colour/native depth and optional 30 fps exposure priority; motor tilt for v1.

Ten automated suites pass. Private replay consistency was checked; private replay data are not part of this repository. Live OVR behavior still needs confirmation. Side views, lying poses and occluded limbs remain limitations of this single-camera system.

## Repository boundaries

Source, tests, build/package scripts, model assets, provenance and licenses are versioned. Recordings, video/images, calibration, sensor identifiers, diagnostics, preferences, caches, Python environments and build/release output are excluded. `scripts/audit_repository.py` checks staged content before publication. Do not upload the whole working folder as an archive.

Third-party model/library terms remain in force. No blanket open-source license is granted for the application code by the third-party notices; this repository is initially private.
