# KinectAIFusionFBT v1.2.0 — Optional trackers

## Changes since v1.1.0

- Add optional left/right knee, left/right elbow and chest trackers to SteamVR or OSC.
- Hips and feet remain the default. Simple **Add: Knees / Elbows / Chest** checkboxes let you choose any combination, up to eight trackers.
- Replaced the confusing layout dropdown and removed the selection-reset behavior that could interfere with choosing an item.
- Extra trackers reuse the existing learned body pose; no additional AI inference. They share output smoothing, freshness checks and playspace conversion.
- Original hips/feet tracking and calibration algorithms remain unchanged.

## Updating an existing installation

1. Close the app and SteamVR completely.
2. Download **KinectAIFusionFBT-v1.2.0-AppUpdate.zip** and extract its contents into your existing v1.1.0 application folder, replacing included files. Your models, calibration and preferences are retained.
3. Run **Update SteamVR Trackers.cmd**, then restart SteamVR and the app.

**Update the app, driver and tracker profile together.** This release uses a new tracker connection format; mixing old and new components will not work.

To add trackers, pause output, tick the desired checkboxes beside Confirm saved alignment, start output, then recalibrate FBT **inside VRChat**. Knees and elbows are pairs. A layout change does not require redoing the app's camera alignment unless that alignment is already wrong.

## New installations

Download all three **KinectAIFusionFBT-v1.2.0-Windows-x64.zip.001 / .002 / .003** parts into one folder. Open `.001` with 7-Zip and extract the complete application. These are archive parts, not separate installers. The full package includes SAM original/optimized models and native runtimes. NVIDIA/Kinect drivers and SteamVR are external prerequisites. SHA-256 checksums are included.

## Validation and known limitations

All ten offline test suites pass, including all tracker combinations, unchanged base outputs, missing-pose handling, and SteamVR/OSC smoothing and playspace parity. The user confirmed additional trackers work. Their live accuracy depends on visibility; torso/limb twist is inferred. Single-camera occlusion, sideways and lying poses remain difficult. The final checkbox UI still needs broader interactive validation.

**Saved alignment after Quest tracking recovery remains unresolved.** Confirm saved alignment can accept an alignment that leaves trackers displaced after headset tracking resumes. Use a fresh **Align to VR** when this occurs. This release does not attempt to fix it.

Packages are assembled from an explicit allowlist and verified by SHA-256 and ZIP CRC. No personal recordings, screenshots, calibration, preferences, diagnostics or GPU caches are included.
