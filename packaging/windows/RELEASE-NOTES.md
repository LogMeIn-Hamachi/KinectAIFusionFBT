# KinectAIFusionFBT v1.2.3-rc.1 — Low-end PC calibration preview

This pre-release lets users with slower or uneven camera processing opt into a more patient VR alignment. In **Advanced**, enable **Low-end PC (calibration)** before clicking **Align to VR**. Each pose can wait longer for steady wrist observations. The ordinary calibration path, final fit thresholds, independent check pose, tracking and GPU cadence are unchanged.

The change addresses a code-level way calibration can remain at 0/12 even with visible wrists. Synthetic 10 fps calibration and moving-wrist rejection tests pass. It has not yet been validated on the user's RTX 4050 or in a live headset session, so this is a test release rather than a proven fix for every 0/12 cause. Diagnostics records whether the option was enabled.

## Existing v1.2.2 installations

1. Close this app and SteamVR. Keep a backup of your existing installation and calibration files.
2. Extract **KinectAIFusionFBT-v1.2.3-rc.1-AppUpdate.zip** into the existing v1.2.2 folder, replacing included files. It includes the matching app and driver, not the unchanged models and native runtimes.
3. Run **Update SteamVR Trackers.cmd** from the updated folder and restart SteamVR.
4. Enable the option in Advanced, then run a fresh **Align to VR** to test it. Check tracker placement before enabling output.

## New installations

Download all three **KinectAIFusionFBT-v1.2.3-rc.1-Windows-x64.zip.001 / .002 / .003** parts into one folder and open `.001` with 7-Zip. Install the appropriate Microsoft Kinect SDK/runtime, NVIDIA driver and SteamVR separately. Follow **START-HERE.html** inside the extracted package.

## Validation and limits

The Release build, all ten offline CTest suites, an isolated app restart test, and the package integrity checks passed. These tests do not establish live tracking accuracy or a particular frame rate on another PC. No personal recordings, calibration, preferences, sensor IDs, diagnostics or GPU caches are included.
