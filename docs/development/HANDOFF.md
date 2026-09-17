# Current handoff — v1.0.1

## Accepted behavior

Native Kinect v1/v2 capture feeds SAM optimized by default: selectively FP8 DINOv3 encoder with TensorRT RTX, GPU-resident features, native CUDA/LibTorch TF32 decoder and optimized sampling. Both encoder and decoder utilize high-priority CUDA streams to prevent preemption by background DirectX GPU graphics loads (e.g. heavy VRChat worlds). An adaptive GPU cadence control (Auto, 30 Hz full, 20 Hz balanced, 15 Hz low GPU) allows throttling AI compute while preserving 30 FPS depth re-registration and foot contacts. SAM original remains available. The model and cadence menus and SDK comparisons are in Advanced.

The SAM body provides articulation; depth supplies metric positioning and independently observed foot surfaces. Camera-led body placement is retained. Two agreeing controllers can apply only a small, slow horizontal correction (bounded 3.5 cm, 1.5 cm/s); there is no continuous headset-to-pelvis position anchoring. Foot orientation remains uncertain when occluded. Do not equate prior uncertainty numbers with measured accuracy.

One controller-only alignment routine learns controller-to-wrist offsets. Every pose waits indefinitely for trigger/Capture pose, then allows three seconds to settle and three seconds of observation. Failed poses preserve prior valid work. A separate holdout checks the result. Proportions have eight seconds to position and four seconds to capture. Head visibility is optional. An in-headset SteamVR visual overlay (`src/vr_overlay.cpp`) renders real-time 3D controller poses, countdowns, wrist tracking progress (`X/12`), and dedicated retry screens with guidance so the user never needs to remove their headset.

Native SteamVR waist/feet output is the default. OSC remains optional. v1.0.1 binds a live standing-to-raw physical reference at alignment/confirmation. Outgoing trackers use that fixed reference; incoming VR devices and interpolated poses are rebased into it. This prevents OVR standing-space movement being cancelled or entering body correction. Ordinary same-universe chaperone edits do not invalidate the reference. A SteamVR restart or changed hardware universe does; a running session then requires Align to VR again. The live reference is intentionally not serialized. After an app restart, reset OVR offsets before confirming saved calibration.

### SteamVR Driver Motion & Rotation Smoothing (v1.0.1)
- **Zero Compositor Velocity**: Outgoing driver velocity is set to zero (`p.vecVelocity = {0,0,0}` and `p.vecAngularVelocity = {0,0,0}`) because SteamVR's compositor extrapolates tracker poses between frames along non-zero velocity, which previously caused rubber-banding and stage-jumping artifacts.
- **Adaptive 1-Euro Monotonic Low-Pass Filter**:
  - **Position**: Cutoff scales monotonically from $f_c = 6.0\text{ Hz}$ at rest (sensor chatter suppression) up to $28.0\text{ Hz}$ under motion ($f_c = 6.0 + 16.0 \times v$). Delivers smooth 90Hz/120Hz/144Hz tracking with sub-12 ms latency and zero overshoot.
  - **Rotation**: Cutoff scales from $f_c = 8.0\text{ Hz}$ at rest to $32.0\text{ Hz}$ when rotating. Incoming orientation uses `targetRot = kf::continuous(targetRot, currentRot_)` to ensure proper quaternion continuity without overwriting the target orientation. A dedicated convergence unit test in `tests/steamvr_output.cpp` guards this.

## Validation boundary

All ten local CTest suites passed. Synthetic tests cover OVR translation/rotation/reset, time-matched pose interpolation, calibration during virtual movement, rotation filter convergence, and full body/contact invariance at 15/22/30 Hz. Live OVR/VRChat confirmation remains necessary; tests are not a claim of measured live accuracy.

Single-camera side views, lying on furniture and limb occlusions remain difficult. Prior user feedback improved after removing aggressive VR anchoring and using user-paced calibration. Do not resurrect those approaches just because historical notes or older scripts mention them.

## Hardware Support
- **GPU**: NVIDIA GeForce RTX 4000 (Ada Lovelace) or RTX 5000 (Blackwell) series required for native hardware FP8 Tensor Cores in `sam3d-optimized`. Tested device: RTX 5070 Ti on Windows 11. Older RTX 20/30 series GPUs lack hardware FP8 instructions and fail TensorRT compilation for FP8 graphs.
- **Sensors**: Kinect for Xbox 360 (v1) with SDK 1.8; Kinect for Xbox One (v2) with SDK 2.0. For Kinect v2 on Windows 11, audio enhancements on the microphone array must be disabled in Sound properties to prevent continuous hardware reboot loops.

## Where to work & Development Workflow

- Capture/exposure: `src/capture.cpp`, `src/capture_v2.cpp`, `src/exposure.cpp`.
- Inference and depth registration: `src/sam3d_model.cpp`, `src/sam3d_torch_decoder.cpp`, `src/sam3d_geometry.cpp`.
- Body continuity and independent foot support: `src/body_tracker.cpp`, `src/estimator.cpp`; `tests/body_tracker.cpp` and `tests/temporal.cpp`.
- Alignment and in-headset overlay: `include/alignment.hpp`, `src/alignment.cpp`, `src/controller_calibration.cpp`, `src/floor_alignment.cpp`, `src/vr_overlay.cpp`, `include/vr_overlay.hpp`; `tests/controller_calibration.cpp`.
- Coordinate transforms/history: `include/core.hpp`, `src/core.cpp`, `src/vr.cpp`.
- Native output and driver smoothing: `include/steamvr_bridge.hpp`, `src/steamvr_driver.cpp`, `src/engine.cpp`; `tests/steamvr_output.cpp`.
- Win32 interface and speech: `src/app.cpp`. Main display is double buffered; avoid unnecessary repeated button label/state changes.
- Deployment: `scripts/prepare_runtime.py`, `scripts/package_distribution.py`, `packaging/windows`, `driver/kinect_fbt`.

### How to Build, Deploy & Test Changes
1. Make code changes in `src/` or `include/`.
2. Build the project: `cmake --build build --config Release`.
3. Run all 10 CTest suites: `ctest --test-dir build -C Release --output-on-failure`. All 10 must pass.
4. Deploy updated binaries to the active test folder (`release/dev/`):
   - `Copy-Item build/Release/driver_kinect_fbt.dll release/dev/steamvr-driver/kinect_fbt/bin/win64/driver_kinect_fbt.dll`
   - `Copy-Item build/Release/KinectRGBD.exe release/dev/KinectRGBD.exe`
   *(Do not wipe or delete the `release/dev/` directory itself, as it houses the models and runtime DLLs).*
5. Run `python scripts/audit_repository.py` before committing. It must report 0 findings.
