# Current handoff — Preview 23

## Accepted behavior

Native Kinect v1/v2 capture feeds SAM optimized by default: selectively FP8 DINOv3 encoder with TensorRT RTX, GPU-resident features, native CUDA/LibTorch TF32 decoder and optimized sampling. SAM original remains available. The model menu and SDK comparisons are in Advanced.

The SAM body provides articulation; depth supplies metric positioning and independently observed foot surfaces. Camera-led body placement is retained. Two agreeing controllers can apply only a small, slow horizontal correction (bounded 3.5 cm, 1.5 cm/s); there is no continuous headset-to-pelvis position anchoring. Foot orientation remains uncertain when occluded. Do not equate prior uncertainty numbers with measured accuracy.

One controller-only alignment routine learns controller-to-wrist offsets. Every pose waits indefinitely for trigger/Capture pose, then allows three seconds to settle and three seconds of observation. Failed poses preserve prior valid work. A separate holdout checks the result. Proportions have eight seconds to position and four seconds to capture. Head visibility is optional.

Native SteamVR waist/feet output is the default. OSC remains optional. Preview 23 binds a live standing-to-raw physical reference at alignment/confirmation. Outgoing trackers use that fixed reference; incoming VR devices and interpolated poses are rebased into it. This prevents OVR standing-space movement being cancelled or entering body correction. Ordinary same-universe chaperone edits do not invalidate the reference. A SteamVR restart or changed hardware universe does; a running session then requires Align to VR again. The live reference is intentionally not serialized. After an app restart, reset OVR offsets before confirming saved calibration.

## Validation boundary

All ten local CTest suites passed. Synthetic tests cover OVR translation/rotation/reset, time-matched pose interpolation, calibration during virtual movement and full body/contact invariance at 15/22/30 Hz. A private 301-frame recorded replay retained identical values across 478 non-timing columns. No private recordings or numeric session exports are included in Git. Live OVR/VRChat confirmation remains outstanding; tests are not a claim of measured live accuracy.

Single-camera side views, lying on furniture and limb occlusions remain difficult. Prior user feedback improved after removing aggressive VR anchoring and using user-paced calibration. Do not resurrect those approaches just because historical notes or older scripts mention them.

## Where to work

- Capture/exposure: src/capture.cpp, capture_v2.cpp, exposure.cpp.
- Inference and depth registration: src/sam3d_model.cpp, sam3d_torch_decoder.cpp, sam3d_geometry.cpp.
- Body continuity and independent foot support: src/body_tracker.cpp, src/estimator.cpp; tests/body_tracker.cpp and temporal.cpp.
- Alignment: include/alignment.hpp; src/alignment.cpp, controller_calibration.cpp, floor_alignment.cpp; tests/controller_calibration.cpp.
- Coordinate transforms/history: include/core.hpp, src/core.cpp, src/vr.cpp.
- Native output: include/steamvr_bridge.hpp, src/steamvr_driver.cpp, src/engine.cpp; tests/steamvr_output.cpp.
- Win32 interface and speech: src/app.cpp. Main display is double buffered; avoid unnecessary repeated button label/state changes.
- Deployment: scripts/prepare_runtime.py, scripts/package_distribution.py, packaging/windows, driver/kinect_fbt.

Older NLF/RTMPose code and experiment scripts remain for development reference, but only the accepted SAM variants are shared as deployed models. Their optional weights and private experiment outputs are absent. The previously shipped Preview 23 ZIP is a runtime package, not this source repository.
