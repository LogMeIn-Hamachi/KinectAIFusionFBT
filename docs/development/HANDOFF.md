# Current handoff — v1.0.1 baseline + private development changes

## Accepted behavior

Native Kinect v1/v2 capture feeds SAM optimized by default: selectively FP8 DINOv3 encoder with TensorRT RTX, GPU-resident features, native CUDA/LibTorch TF32 decoder and optimized sampling. Both encoder and decoder use high-priority CUDA streams; graphics contention can still affect inference. Neural cadence control (Auto, 30 Hz full, 20 Hz balanced, 15 Hz low GPU) limits new image inference while depth re-registration and foot contact processing continue on every delivered camera frame. Kinect v2 synchronized capture can deliver only 15 fps in dim light; this is not an independent guaranteed 30 Hz depth processing path. SAM original remains available. The model and cadence menus and SDK comparisons are in Advanced.

The SAM body provides articulation; depth supplies metric positioning and independently observed foot surfaces. Camera-led body placement is retained. Two agreeing controllers can apply only a small, slow horizontal correction (bounded 3.5 cm, 1.5 cm/s); there is no continuous headset-to-pelvis position anchoring. Foot orientation remains uncertain when occluded. Do not equate prior uncertainty numbers with measured accuracy.

One controller-only alignment routine learns controller-to-wrist offsets. Every pose waits indefinitely for trigger/Capture pose, then allows three seconds to settle and three seconds of observation. Failed poses preserve prior valid work. A separate holdout checks the result. Proportions have eight seconds to position and four seconds to capture. Head visibility is optional. An in-headset SteamVR visual overlay (`src/vr_overlay.cpp`) renders real-time 3D controller poses, countdowns, wrist tracking progress (`X/12`), and dedicated retry screens with guidance so the user never needs to remove their headset.

Native SteamVR waist/feet output is the default. OSC remains optional. v1.0.1 binds a live standing-to-raw physical reference at alignment/confirmation. Outgoing trackers use that fixed reference; incoming VR devices and interpolated poses are rebased into it. This prevents OVR standing-space movement being cancelled or entering body correction. Ordinary same-universe chaperone edits do not invalidate the reference. A SteamVR restart or changed hardware universe does; a running session then requires Align to VR again. The live reference is intentionally not serialized. After an app restart, reset OVR offsets before confirming saved calibration.

### SteamVR Driver Motion & Rotation Smoothing (v1.0.1)
- **Zero Compositor Velocity**: Outgoing driver velocity is set to zero (`p.vecVelocity = {0,0,0}` and `p.vecAngularVelocity = {0,0,0}`) because SteamVR's compositor extrapolates tracker poses between frames along non-zero velocity, which previously caused rubber-banding and stage-jumping artifacts.
- **Adaptive 1-Euro Monotonic Low-Pass Filter**:
  - **Position**: Cutoff scales monotonically from $f_c = 6.0\text{ Hz}$ at rest (sensor chatter suppression) up to $28.0\text{ Hz}$ under motion ($f_c = 6.0 + 16.0 \times v$). The fixed-target response is monotonic. No end-to-end latency or live jitter improvement is claimed from the offline tests.
  - **Rotation**: Cutoff scales from $f_c = 8.0\text{ Hz}$ at rest to $32.0\text{ Hz}$ when rotating. Incoming orientation uses `continuous(targetRot, rotation_)` in the shared `TrackerSmoothing` helper to ensure proper quaternion continuity without overwriting the target orientation. A dedicated convergence unit test in `tests/steamvr_output.cpp` guards this.

## Validation boundary

All ten local CTest suites passed. Synthetic tests cover OVR translation/rotation/reset, time-matched pose interpolation, calibration during virtual movement, rotation filter convergence, and full body/contact invariance at 15/22/30 Hz. Live OVR/VRChat confirmation remains necessary; tests are not a claim of measured live accuracy.

Single-camera side views, lying on furniture and limb occlusions remain difficult. Prior user feedback improved after removing aggressive VR anchoring and using user-paced calibration. Do not resurrect those approaches just because historical notes or older scripts mention them.

## Private development — 2026-09-17: cadence and SteamVR stability

These changes follow the accepted v1.0.1 baseline and await user testing. Publish only to the private development repository until the user explicitly authorizes a public release.

- `include/neural_cadence.hpp` replaces frame-modulo throttling with timestamp deadlines. 20/15 Hz are rate ceilings: a 15 fps camera is no longer divided down to 10/7.5 Hz. Replay still infers every frame. Selection/sensor/mode changes reset scheduling history.
- Auto uses **fresh inference-frame worker cost** (registration, inference and body/estimator processing) and actual queue wait, excluding capture conversion time. It does not measure GPU utilization or VR compositor headroom. The first two inference samples are excluded for warmup. A 350 ms exponential average and 600 ms sustained pressure lower the ceiling one step; two seconds of sustained recovery raise it one step. Thresholds: down from 30 at >34.5 ms work or >18 ms queue; down from 20 at >50.5 ms or >30 ms queue. Recover from 15 below 46/14 ms and from 20 below 31.5/10 ms. These are initial policy thresholds, validated offline rather than tuned against current live VRChat load.
- This fixes the old 15 Hz latch: moderate load repeatedly renewed the low-rate countdown without allowing recovery to 20. Skipped frames no longer retrigger a decision using stale inference timing.
- `include/tracker_smoothing.hpp` is used by both the real driver and regression tests. Position/angular derivatives come from changes between new source observations, with signed-vector smoothing. They no longer use filter error divided by compositor frame time. Protocol v2 `validUntil - outputHoldSeconds` supplies the observation timestamp, so the ABI is unchanged. Position/rotation cutoff ranges and zero compositor velocities are retained. Filter reset and update now share the pose mutex.
- `bridgeTrackingStatus` separates device connection from pose validity. A temporarily hidden/expired foot reports connected but out of range while the enabled writer is live. It still becomes invalid immediately when its observation expires; pausing output or losing the writer disconnects it. This fixes a code path that previously reported a hot unplug on every pose loss. It does not establish the cause of the user's suspected live dropouts.
- GUI displays actual completed-inference Hz beside the selected ceiling. Diagnostics distinguish inference runs, reused frames, exceptions, unavailable body priors, per-tracker source changes and validity losses, and successfully published invalid-pose transitions while the driver is live. Those last counters are **not driver acknowledgements**. Numeric counters reset on Start; selection changes reset transition baselines. No automatic camera recording or telemetry was added. `neural_result_age_ms` is time since inference completion, not image exposure age or motion-to-photon latency.
- Existing cached-pose extrapolation, camera-led articulation, controller calibration, independent foot contacts, tracking expiration limits and OVR coordinate conversions are unchanged. Headset anchoring has not been reintroduced.

Validation: full Release build and all ten CTest suites pass. Added synthetic coverage for 15/20/30 fps camera/rate combinations, startup spikes, sustained load and recovery, backlog recovery, replay, mode/clock changes, health counters, 90/120/144 Hz driver derivatives, quaternion sign flips, convergence, reacquisition, and connected-but-untracked poses. No live camera, recording or VR output was started. Test success is not a live accuracy or GPU-load measurement.

## Private development — 2026-09-17: OSC output parity

OSC remains the secondary output. `include/osc_tracking.hpp` now uses the same `trackerPacket` bounded prediction (up to 100 ms total), source validity deadlines and `TrackerSmoothing` implementation as the native driver. Previously OSC had only the estimator smoothing and at most 40 ms output prediction. Smoothing occurs in the fixed physical reference; the live raw-to-standing playspace transform is applied afterward, followed once by the existing Unity reflection and ZXY Euler serialization. No head OSC messages or head anchoring were added.

The existing output loop sends at roughly 125 Hz before work/scheduling overhead, independent of neural cadence. OSC is not synchronized to the compositor. It retains nonblocking localhost UDP and one coherent immediate bundle of position/rotation pairs with stable tracker IDs. Invalid trackers are omitted individually and their filter history resets. Pause, invalid live reference, calibration/reference or player changes reset filtering. Legacy OSC-only calibrations without a raw reference remain supported. Native output behavior and the default output selection are unchanged.

OSC status now shows valid tracker count, missing-reference waits and send failures. Output validity counters also cover OSC sender-side transitions; successful UDP sends do not acknowledge VRChat reception. The documented [VRChat OSC tracker API](https://docs.vrchat.com/docs/osc-trackers) provides position/rotation but no explicit per-tracker validity/disconnect message. We stop refreshing expired poses; VRChat controls receiver timeout and IK behavior, so this cannot duplicate SteamVR's immediate invalid-pose reporting.

Validation: Release build and all ten CTest suites pass. The SteamVR output suite compares the actual OSC helper against native prediction/filtering for 15/30 Hz camera samples at 125 Hz delivery, including rotations, head gaze changes, OVR translation/rotation/reset, independent foot expiration, lost-reference rejection, recalibration and legacy OSC alignment. Existing core tests cover OSC wire encoding, stable addresses, units/handedness and malformed-pose rejection. No live OSC or tracker output was enabled; receiver-side feel still needs user validation.

## Complete Chronological Changelog of Improvements (Preview 23 to v1.0.1)

1. **Adaptive GPU Cadence (`ca27b01`, `4d1041d`, `e69b745`)**:
   - Added multi-tier neural cadence control (`Auto`, `30 Hz full`, `20 Hz balanced`, `15 Hz low GPU`) in `src/engine.cpp` and `src/body_tracker.cpp`.
   - Allows throttling neural network compute during heavy VRChat GPU loads while re-registering depth and foot contacts on each delivered capture frame.
   - Assigned dedicated high-priority CUDA compute streams (`c10::cuda::getStreamFromPool(true)`) to both TensorRT and LibTorch for neural compute scheduling. This does not guarantee freedom from graphics contention.
2. **Torch Decoder Tensor Contiguity (`0f0afbb`)**:
   - Added `.contiguous()` before raw pointer `std::memcpy` extraction in `src/sam3d_torch_decoder.cpp`, preventing corrupted joint output from strided tensor slices.
3. **Camera Exposure Automation (`2761248`)**:
   - Removed legacy manual "prioritize 30 fps" checkbox that caused underexposure; camera exposure is automatic by default across both Kinect v1 and v2.
4. **In-Headset SteamVR Visual Calibration Overlay (`4815046`, `a9b549a`, `ba2d7dd`)**:
   - Built an in-headset OpenVR dashboard/world overlay (`src/vr_overlay.cpp`, `include/vr_overlay.hpp`).
   - Renders live 3D visual guide poses, countdowns, wrist tracking progress (`X/12`), and dedicated retry screens with guidance so the user never needs to remove their headset.
   - Relaxed Pose 5 ergonomics to shoulder-width forward reach clear of the torso.
   - Forwarded learned wrist prior to holdout validation to eliminate false calibration failures.
5. **SteamVR Driver Monotonic Smoothing (`6e97083`)**:
   - Eliminated the "move > jump > freeze > jump" rubber-banding and stage-jumping artifacts.
   - Set compositor velocity to zero (`p.vecVelocity = {0,0,0}` and `p.vecAngularVelocity = {0,0,0}`) to stop SteamVR's compositor from extrapolating tracker poses ahead in time between frames.
   - Implemented an adaptive 1-Euro monotonic low-pass filter for position ($f_c = 6.0\text{ Hz}$ to $28.0\text{ Hz}$) with monotonic convergence to a fixed target; end-to-end latency was not established by the convergence test.
6. **Hotfix: Restored Tracker Rotation (`3e31704`)**:
   - Fixed an inverted argument order in `targetRot = kf::continuous(targetRot, currentRot_)` in `src/steamvr_driver.cpp` that had previously overwritten target rotation with current rotation every frame, restoring full 360° rotation for feet and body turns.
   - Added 1-Euro adaptive rotation filter ($f_c = 8.0\text{ Hz}$ to $32.0\text{ Hz}$).
   - Added automated rotation convergence unit test in `tests/steamvr_output.cpp`.
7. **Hardware Requirements & Troubleshooting Guidance (`e28ed87`)**:
   - Documented NVIDIA GeForce RTX 4000 / 5000 series requirement due to native hardware FP8 Tensor Cores in `sam3d-optimized`.
   - Documented Windows 11 Kinect v2 microphone array audio enhancement reboot loop fix.
8. **Public Release & Standalone Packaging (`0678611`, `3e31704`, Releases v1.0.0 & v1.0.1)**:
   - Established public GitHub repository: `https://github.com/LogMeIn-Hamachi/KinectAIFusionFBT`.
   - Published standalone release packages with 3-part split archives and drop-in `AppUpdate.zip`.

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

## Private development — SteamVR driver updater

`packaging/windows/Update SteamVR Trackers.cmd` launches `Update-SteamVR-Trackers.ps1`. Run it from a complete, permanently located package with SteamVR closed. It identifies older Kinect registrations by manifest name, switches registration using SteamVR's own vrpathreg utility, verifies the result, and attempts rollback on failure. It does not download or overwrite driver binaries, delete old folders/settings, or modify unrelated driver registrations. SteamVR loads the new package's bundled DLL on next start. Same-folder retries are no-ops. The normal installer now treats its already registered folder as success and directs conflicting installations to the updater.

Both runtime assembly and the distribution allowlist include the updater. Run `powershell.exe -NoProfile -ExecutionPolicy Bypass -File tests/driver_update.ps1` for offline fixture tests (new/same folder, first install, duplicates, running-SteamVR refusal, incomplete package, failure rollback, unrelated-driver preservation). These mock registration calls and never alter the real SteamVR installation. No live registration change was performed during implementation.
