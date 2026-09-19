# KinectAIFusionFBT v1.2.2 — Saved alignment and reliability

## Changes since v1.2.0

- Save the complete calibration coordinate reference. Restore it across app/capture restarts and playspace changes without attaching old coordinates to a new standing origin. SteamVR and OSC share the repair.
- Permit explicit same-headset/connection restoration after a runtime change. Reject incompatible or unavailable references; confirmation never starts output automatically.
- Preserve captured proportions across temporary tracking gaps and verified recovery of the same person.
- Harden SteamVR input/overlay shutdown and app error recovery. Failed calibration saves now retain session usability with a clear warning.
- Make calibration progress follow actual accepted observations and clarify diagnostic/replay controls.
- Preserve coordinate references and captured proportions in new recordings, while retaining readers for older recordings.
- Reuse the SAM image input buffer without changing input pixels. This reduces allocation/preprocessing work; no claim of increased VRChat frame rate.
- Add local numeric processing and VR-reference diagnostics, plus version/source identification.

Tracking articulation, foot contact, smoothing and Auto/30/20/15 Hz policies retain their existing behavior. Optional knees, elbows and chest remain available; default is hips and feet.

## Upgrade instructions

1. Close the app and SteamVR.
2. Extract **KinectAIFusionFBT-v1.2.2-AppUpdate.zip** into an existing v1.2.0 installation, replacing the included files. Models and native dependencies are unchanged. Personal calibration/preferences are not included in the update.
3. Run **Update SteamVR Trackers.cmd**, then restart SteamVR and the app.
4. **Run Align to VR once after upgrading.** Old saves lack the original coordinate reference. Learned wrist offsets are retained, but cannot supply the missing reference.

Update app and driver together. New recordings require this version's reader; older recordings remain readable. Back up your installation before downgrading because older versions cannot read the new saved-alignment format.

## New installations

Download all three **KinectAIFusionFBT-v1.2.2-Windows-x64.zip.001 / .002 / .003** parts into one folder and extract `.001` with 7-Zip. All parts are required. The full package includes SAM original/optimized and native runtimes. Kinect/NVIDIA drivers and SteamVR are external prerequisites. Checksums are included.

## Validation and remaining limitations

Full Release build, all ten offline suites, offline startup/restart checks and installer/updater fixture tests passed. Saved-reference tests cover restart, combined playspace translation/rotation, incoming controllers and all eight SteamVR/OSC tracker roles. Packages use explicit file allowlists, model hashes and ZIP integrity checks.

**Full Quest sleep/resume recovery still needs live verification.** A short numeric headset-removal test did not establish a full sleep transition or physical tracker alignment afterward. This release fixes demonstrated persistence defects; it does not guarantee recovery from an unreported headset-map shift. Check placement after confirming, and align again after moving the Kinect or changing physical room setup. Diagnostics saves numbers only, not camera images.

Single-camera occlusion, sideways and lying poses remain difficult. Overlay reliability still needs broader live validation. No personal media, calibration files, preferences, diagnostics, device identifiers or GPU caches are distributed.
