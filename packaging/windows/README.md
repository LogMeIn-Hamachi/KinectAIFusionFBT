# KinectAIFusionFBT — v1.2.3-rc.1 pre-release

Full-body tracking for SteamVR using Kinect v1 or v2 and Fast SAM 3D Body. Native Windows x64; NVIDIA GeForce RTX 4000 or 5000 series GPU required (tested on Windows 11 with RTX 5070 Ti). Older RTX 20/30 series lack hardware FP8 and are not supported.

Open **START-HERE.html** for the setup guide. Extract the entire ZIP into a writable folder before running **KinectRGBD.exe**. Keep its files together. This is a full package, not an update patch: both SAM models and their native runtimes are included.

This preview adds **Low-end PC (calibration)** in Advanced. Enable it before Align to VR if wrist counts remain at 0/12 on a slower PC. It allows more time for steady samples; final alignment checks and normal tracker output are unchanged.

## First use

1. Install the official Microsoft Kinect runtime/SDK for your sensor and the NVIDIA display driver. Kinect v2 needs its powered USB 3 adapter. The SDK 2.0 installation also supplies the v2 runtime; an existing working installation is sufficient. Links are in START-HERE.html.
2. Install and run SteamVR once. Close SteamVR, double-click **Install SteamVR Trackers.cmd**, then restart SteamVR. SteamVR may ask you to enable the Kinect tracker add-on. Keep the package at this location; registration refers to this folder.
3. Open **KinectRGBD.exe**, press **Start**, choose your body and **Lock player**. First use prepares a local GPU cache and takes longer than normal startup. SAM optimized is the default. You can stop tracking and choose SAM original in **Advanced** if needed.
4. Set camera tilt before alignment. Keep wrists and feet visible. Kinect v1 has motor buttons; v2 must be adjusted by hand. A visible head is optional.
5. Press **Capture proportions**. You have eight seconds to get into place before a four-second capture. Stand naturally with your feet apart.
6. Reset OVR Advanced Settings space-drag/rotation offsets. Press **Align to VR**. Each pose waits for you: squeeze either controller trigger or click **Capture pose** when ready, then stay still after the three-second countdown. Keep your normal grip and follow the hand-position guide: waist, apart, chest, forward reach, then left high/right low. Failed poses can be retried without restarting the earlier poses.
7. Click **Start trackers**. In SteamVR, use Waist, Left Foot and Right Foot roles; then use VRChat's FBT calibration.

## Everyday use and playspace movement

Run **Align to VR once after upgrading**: older saves lack the original coordinate reference. New saves preserve it across app restarts and playspace movement. Keep the Kinect, physical room and headset connection unchanged, then Start, Lock player and **Confirm saved alignment**. After a SteamVR restart, let tracking settle and confirm again. Check placement before enabling output.

OVR space drag and rotation should move all selected trackers with the headset/controllers. Reset OVR offsets before a fresh alignment; new saved references do not need rebinding to today's standing origin. A moved Kinect, new room setup or unreported headset-map shift still needs fresh alignment.

## Choose extra trackers

Hips and feet are always included. Beside Confirm saved alignment, tick **Add: Knees**, **Elbows**, or **Chest** in any combination. Knees and elbows each add both sides. Leave all unchecked for the default three trackers. Pause tracker output before changing these choices, then start output and recalibrate FBT inside VRChat. The selection is remembered. SteamVR and OSC both support the extras, using the existing AI estimate without another inference.

Extra joint positions and rotations are estimates; visibility and occlusion affect quality. Existing hips/feet tracking and smoothing are retained.

**Validation limit:** saved-reference persistence is repaired and covered by offline regression tests. Full Quest sleep/resume recovery has not yet been verified. If trackers remain displaced, click Diagnostics and run Align to VR before using output.

## What is included

- SAM optimized: selectively FP8 image encoder, native CUDA TF32 decoder with optimized sampling, shared GPU intermediate data.
- SAM original: original encoder/decoder assets as an alternative. Both require a supported NVIDIA GPU; original does not provide a CPU fallback. Other GPU configurations are unverified.
- Native SteamVR tracker driver. OSC output, diagnostic SDK comparisons and model selection are available in Advanced.
- Native NVIDIA, ONNX Runtime, PyTorch and Microsoft VC++ runtime libraries; third-party license texts in docs/licenses.

No Python or CUDA developer toolkit is needed to run this package. Kinect drivers, SteamVR and the NVIDIA display driver are installed separately. Model caches are generated locally; several additional GB of writable disk space may be needed during first-run preparation.

## Privacy and maintenance

Camera processing stays on your PC. Recording is opt-in using Record locally and can consume substantial disk space. This distribution contains no recordings, camera images, calibration, preferences, diagnostic exports, or prebuilt local cache. Back up your calibration files before replacing your own installation.

To move the app: close it, close SteamVR, run **Remove SteamVR Trackers.cmd** in the old folder, move the complete folder, and run **Install SteamVR Trackers.cmd** there. Restart SteamVR. To uninstall, remove the SteamVR registration first, then delete the extracted folder after keeping any recordings/settings you want.

Read **docs/TROUBLESHOOTING.md** for common problems and **THIRD_PARTY_NOTICES.md** for component terms. Converted Meta/DINOv3 models retain their respective agreements; these third-party terms are included in the package.

## Update the SteamVR driver

1. Close SteamVR completely.
2. Extract the new package to a permanent folder.
3. Double-click **Update SteamVR Trackers.cmd** in the new folder, then start SteamVR.

This switches SteamVR to the driver bundled in this package, replacing older Kinect registrations without deleting old files or settings. If this folder is already registered, no registration change is needed: SteamVR loads the current files here on startup. The updater does not download updates or transfer preferences. Keep the new folder in place. Failed registration changes attempt to restore the previous Kinect registration.
