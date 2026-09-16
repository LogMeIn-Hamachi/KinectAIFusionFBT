# Agent handoff

Read README.md, docs/development/BUILD.md and docs/development/HANDOFF.md before changing tracking. This is a native Windows C++20 project; no particular AI editor or service is required. The accepted baseline is Preview 23.

- Preserve camera-led SAM articulation, independent depth-supported foot contact, and the small bounded horizontal controller correction. Continuous headset-to-hip anchoring was deliberately removed. Head gaze must not move feet.
- Calibration is controller-only, learns wrist offsets, waits for user readiness on every pose and validates against a separate holdout pose. Proportions have an eight-second preparation countdown. Do not restore timed pose switching or headset-specific presets.
- OVR playspace changes are virtual coordinate transforms, not physical observations. Keep the live calibration reference, VR sample interpolation, incoming constraints and outgoing tracker conversion consistent. Check tests/steamvr_output.cpp and tests/body_tracker.cpp.
- Use targeted offline tests first. The ten CTest suites require no camera, recording or live tracker output. Coordinate any live test with the user; recording camera data requires their explicit permission. Never enable VR output merely to test startup.
- Do not commit recordings, screenshots/video, calibration, sensor IDs, diagnostics, GPU caches, virtual environments, tokens, user preferences or build/release folders. Keep the allowlist in .gitignore and run scripts/audit_repository.py before publishing.
- Models and the vendored library use Git LFS. Never replace missing model bytes with empty files or commit the weights as ordinary Git blobs. Keep model integrity manifests and upstream license texts.
- Do not treat legacy analysis scripts as current instructions. Archived experiments refer to local artifacts that are intentionally absent from Git. State what was actually measured; replay consistency is not live tracking accuracy.
- Do not delete or overwrite the user's working release to test a build. Assemble a fresh release/dev folder with scripts/prepare_runtime.py. Keep build changes and tracking changes separate when possible.
- Update the handoff and tests when behavior changes. No autonomous delegation is required by this file.
