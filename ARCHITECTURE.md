# Architecture

The application is native Win32/C++20. Capture, inference/estimation, output, recording and UI use separate work paths with bounded queues. Kinect v1 and v2 are loaded through the installed Microsoft runtimes. The desktop app does not use Python or cloud inference.

## Data flow

1. Native colour, depth, player/body index, skeleton/floor and timestamps form a Frame. Kinect SDK data supports association and explicit fallback.
2. A selected-person crop feeds the SAM encoder through ONNX Runtime/TensorRT RTX. The encoder writes its feature tensor into a decoder-owned CUDA buffer; both stages share a CUDA stream.
3. The native LibTorch decoder uses TF32 and a captured CUDA graph. SAM geometry is associated with Kinect's metric depth and independent foot surfaces.
4. BodyTracker maintains camera-led articulation and bounded temporal continuity. Two agreeing controllers may contribute a small, slow horizontal correction. There is no continuous headset-to-pelvis anchor.
5. Estimator produces waist and foot tracker poses and optional knees, elbows and chest, with independent foot-contact stabilization and bounded prediction. Optional roles consume the same learned pose without feeding back into the base trackers. Engineering uncertainty priors are not measured accuracy.
6. Native SteamVR packets flow through shared memory to the packaged driver. SteamVR and OSC share prediction/smoothing; their final coordinate conversion and receiver behavior differ.

## Coordinate contract

Kinect camera coordinates and calibrated VR coordinates are distinct. Proper rigid transforms handle alignment; Unity OSC uses its required reflection/Euler convention while native SteamVR does not.

Calibration.transform maps camera points into a fixed reference standing space. Calibration.standingToRaw maps that reference into raw tracking space and is captured at alignment. Saved calibration format 2 retains both transforms, a local headset/streamer identity key and the observed universe ID. Confirmation preserves the original matrix and adopts only the current live epoch; it never rebinds old standing coordinates to the current matrix. Legacy calibration files retain offsets but require one new alignment. Recording format 3 stores both calibrated and per-frame matrices/epochs for offline reproduction; the new source/universe/event metadata is live-only in recordings and defaults to zero on both sides during replay.

Each incoming VrSample supplies its current standingToRaw transform. Samples are rebased to one physical reference before interpolation and before controller constraints. Native output uses the fixed reference-to-raw transform; SteamVR applies its current playspace offset exactly once. OVR changes must not become apparent body motion or be cancelled on output. OSC uses the corresponding current standing transform.

A restart or nonzero universe change pauses live output. Explicit restoration requires a recent settled reference from the same headset connection, while preserving the original calibration. Epochs and universe IDs are not durable room identity. Same-universe virtual standing/seated resets and chaperone commits do not invalidate alignment; a commit can be an OVR edit. An observed RoomSetupStarting event invalidates the saved alignment. Source changes are interpolation barriers. No automatic output restart or HMD-motion correction is introduced. Unreported physical remaps still require new calibration; the application does not access the separate headset driver-to-world transform.

VrReferenceHistory retains up to 600 numeric snapshots in memory, normally four per second with extra entries for events, transforms and validity changes. It survives capture restarts and writes only when Diagnostics is requested. Source keys are local identity hashes, not raw serial strings or proof of physical alignment. Confirmation uses the latest poll, with the camera reference caught up, rather than a stale camera frame's VR state.

## Calibration and privacy

Controller-only alignment learns wrist offsets from diverse steady observations. Each pose waits for a trigger/button, then provides a settling countdown and capture interval. A distinct holdout checks the fit. Body proportions have a separate eight-second preparation and four-second capture.

Recording is opt-in, local and potentially very large. Replay does not enable live output. Calibration, diagnostics, camera media, cached inference and user preferences never belong in Git. Runtime assembly and distribution use explicit file selection rather than copying a user's whole release directory.

## Validation

Core tests cover geometry, time response, source continuity, body/foot constraints, calibration, exposure retry logic, recording format and SteamVR coordinate/packet behavior. They are suitable for CPU-only CI. Native inference and live camera/VR tests are separate and require a suitable machine and coordinated user participation. See docs/development/HANDOFF.md for the precise current validation boundary.

Replay preserves recorded geometry settings and permits explicit comparison-mode/output-layout choices. It still infers on every frame and is not a reproduction of live adaptive cadence or GPU contention. Runtime input/overlay calls share a lifetime guard; initialization and shutdown wait for in-flight calls without serializing ordinary input and rendering.
