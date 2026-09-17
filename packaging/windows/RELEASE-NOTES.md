# KinectAIFusionFBT v1.1.0

## Changes since v1.0.1

- Improved automatic neural cadence recovery and tracker continuity during brief tracking loss.
- More continuous slow tracker movement between camera observations.
- OSC now shares prediction and smoothing with native SteamVR output as closely as its protocol allows.
- Saved alignment survives temporary tracking loss and cancelled calibration attempts, while genuine origin changes still require realignment.
- Five self-paced calibration poses with a body-and-hands headset guide, clearer preparation/capture timing and retry feedback.
- Headset overlay uses a persistent GPU texture, corrected GPU submission ordering and VR-session recovery. Diagnostics include overlay errors.
- Added Update SteamVR Trackers.cmd to switch an existing installation to the new bundled driver.

## Installation and updating

New users: download all three Windows-x64.zip parts into one folder, then open .001 with 7-Zip and extract the complete application.

Existing v1.0.0/v1.0.1 users: close the app and SteamVR, extract AppUpdate.zip into the existing application folder and replace the included files. Models, calibration and preferences are retained. Run Update SteamVR Trackers.cmd from that folder, then start SteamVR and the app.

The full package includes both SAM model variants and native runtimes. NVIDIA and Microsoft Kinect drivers and SteamVR remain external prerequisites.

## Validation and limitations

All ten offline CTest suites pass; the updater also has offline fixture tests. Packages are assembled from an explicit allowlist and verified against SHA-256 hashes. Camera recordings, calibration, preferences, diagnostics and GPU caches are excluded.

Recent overlay recovery and the revised poses have offline coverage but still need broader live headset validation. Single-camera occlusion, side-on and lying poses remain difficult. This release does not claim those limitations are solved.
