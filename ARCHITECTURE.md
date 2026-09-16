# Architecture — Preview 23

The application is native Win32/C++20. Capture, inference/estimation, output, recording and UI use separate work paths with bounded queues. Kinect v1 and v2 are loaded through the installed Microsoft runtimes. The desktop app does not use Python or cloud inference.

## Data flow

1. Native colour, depth, player/body index, skeleton/floor and timestamps form a Frame. Kinect SDK data supports association and explicit fallback.
2. A selected-person crop feeds the SAM encoder through ONNX Runtime/TensorRT RTX. The encoder writes its feature tensor into a decoder-owned CUDA buffer; both stages share a CUDA stream.
3. The native LibTorch decoder uses TF32 and a captured CUDA graph. SAM geometry is associated with Kinect's metric depth and independent foot surfaces.
4. BodyTracker maintains camera-led articulation and bounded temporal continuity. Two agreeing controllers may contribute a small, slow horizontal correction. There is no continuous headset-to-pelvis anchor.
5. Estimator produces waist and foot tracker poses, validity, velocities and engineering uncertainty priors, with independent foot-contact stabilization and bounded prediction. These priors are not measured accuracy.
6. Native SteamVR packets flow through shared memory to the packaged driver. OSC is an optional separate coordinate conversion/output path.

## Coordinate contract

Kinect camera coordinates and calibrated VR coordinates are distinct. Proper rigid transforms handle alignment; Unity OSC uses its required reflection/Euler convention while native SteamVR does not.

Calibration.transform maps camera points into a fixed reference standing space. Live Calibration.standingToRaw maps that reference into raw physical tracking space. It is captured at alignment/confirmation, carries a runtime epoch and is never serialized into old calibration or recording formats.

Each incoming VrSample supplies its current standingToRaw transform. Samples are rebased to one physical reference before interpolation and before controller constraints. Native output uses the fixed reference-to-raw transform; SteamVR applies its current playspace offset exactly once. OVR changes must not become apparent body motion or be cancelled on output. OSC uses the corresponding current standing transform.

A restart or hardware-universe change invalidates a live reference. Same-universe virtual chaperone edits do not. After restarting the app, reset OVR offsets before confirming a saved alignment. If SteamVR restarts during tracking, perform Align to VR again.

## Calibration and privacy

Controller-only alignment learns wrist offsets from diverse steady observations. Each pose waits for a trigger/button, then provides a settling countdown and capture interval. A distinct holdout checks the fit. Body proportions have a separate eight-second preparation and four-second capture.

Recording is opt-in, local and potentially very large. Replay does not enable live output. Calibration, diagnostics, camera media, cached inference and user preferences never belong in Git. Runtime assembly and distribution use explicit file selection rather than copying a user's whole release directory.

## Validation

Core tests cover geometry, time response, source continuity, body/foot constraints, calibration, exposure retry logic, recording format and SteamVR coordinate/packet behavior. They are suitable for CPU-only CI. Native inference and live camera/VR tests are separate and require a suitable machine and coordinated user participation. See docs/development/HANDOFF.md for the precise current validation boundary.
