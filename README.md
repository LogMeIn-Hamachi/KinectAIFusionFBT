# KinectAIFusionFBT

Kinect full-body tracking for **SteamVR and VRChat**, using Fast SAM 3D Body and Kinect depth. Runs locally on your PC. SteamVR trackers are the primary output; VRChat OSC is also supported.

**[Download the latest release](https://github.com/LogMeIn-Hamachi/KinectAIFusionFBT/releases/latest)**

## How the tracking works

**AI pose estimation with real depth:** Fast SAM 3D Body estimates your body pose from the colour camera. Kinect depth helps place that pose in physical space and supports foot contact with the floor. The AI drives body articulation, while a small, bounded controller correction helps horizontal alignment. Headset gaze does not drive your hips or feet.

**NVIDIA acceleration:** The native C++20 app runs the optimized SAM image encoder through TensorRT with selective FP8 precision, and its body decoder on CUDA with TF32 acceleration. Intermediate model data stays on the GPU to reduce transfers. Adaptive neural refresh balances new AI estimates against processing load; depth and foot-contact processing continue on delivered camera frames between estimates.

**Smooth tracker output:** An adaptive 1-Euro filter reduces small positional and rotational jitter while responding to movement. SteamVR and OSC share the smoothing implementation. OVR playspace movement is applied as a coordinate transform so virtual space dragging does not become a physical body movement.

All camera and model processing stays on your PC. Tracking quality and responsiveness still depend on visibility, camera frame rate and available GPU time.

## Requirements

- **Windows 10/11 x64** and an **NVIDIA GeForce RTX 4000 or 5000 series GPU**. The current accelerated package does not support older GPUs or CPU-only tracking.
- **Kinect v1 / Xbox 360** with its powered USB adapter, or **Kinect v2 / Xbox One** with its powered USB 3.0 adapter. Connect v2 directly to a USB 3.0 port.
- **SteamVR**, a compatible headset, and **two positional VR controllers** for alignment.
- The official Microsoft SDK: **[SDK 1.8 for Kinect v1](https://www.microsoft.com/en-us/download/details.aspx?id=40278)** or **[SDK 2.0 for Kinect v2](https://www.microsoft.com/en-us/download/details.aspx?id=44561)**, plus an up-to-date NVIDIA driver.

## Install and start

1. Download all three parts of the full Windows package from **Releases**. Open `.zip.001` with 7-Zip and extract to a permanent folder.
2. With SteamVR closed, run **Install SteamVR Trackers.cmd** from that folder.
3. Start SteamVR and **KinectRGBD.exe**, then click **Start**. The first model load prepares a GPU cache and can take a while.
4. Position the camera so it sees your feet, floor and wrists. Your head does not need to be visible. Select your body and click **Lock player**. Kinect v1 has motor tilt buttons; adjust v2 by hand.
5. Click **Capture proportions**. Use the eight-second countdown to stand comfortably with feet shoulder-width apart; hold still during the four-second capture.
6. Reset any OVR space-drag offsets, then click **Align to VR**. Follow the five poses shown in your headset. Keep a normal controller grip and wrists clear of your body. Each pose waits for your trigger press and gives three seconds to settle; retries stay on the same step.
7. Click **Start trackers**, then perform **VRChat's FBT calibration** to fit your avatar.

Single-camera tracking still depends on visibility: hidden limbs, side-on poses and lying down can be less reliable.

## Updates and everyday use

**Update:** Close the app and SteamVR. Extract the release's **AppUpdate.zip** into your existing installation when its release notes say it is compatible, or extract the full package to a permanent folder. Run **Update SteamVR Trackers.cmd** from the updated package, then restart SteamVR. Keep the app and bundled driver versions together.

**Saved alignment:** With the same Kinect position, room and headset connection, start tracking, lock your player and use **Confirm saved alignment** once VR tracking settles. Check tracker placement before starting output. Saves from before v1.2.2 need one fresh alignment. Confirmation does not measure your physical position again; a changed room reference or moved camera needs **Align to VR**. Full Quest sleep/resume recovery has not yet been verified.

**Extra trackers:** Hips and feet are the default. The **Knees / Elbows / Chest** checkboxes add optional trackers, up to eight total. Stop output before changing them, then repeat VRChat's FBT calibration. Occluded limbs and joint rotations remain approximate.

**GPU load:** Leave neural refresh on **Auto**, or choose **20 Hz / 15 Hz** in Advanced to reduce AI work. **30 Hz** requests the highest rate, subject to camera speed and PC load.

**OSC:** Enable OSC in VRChat and select OSC output in the app. It shares the app's smoothing approach with SteamVR, although VRChat's receiving and tracking-loss behavior differs.

## Troubleshooting

### Kinect stopped working after D.R.A.F.T. VR / libusbK installation

If D.R.A.F.T. VR's driver setup (or Zadig) replaced the Kinect drivers with **libusbK**, Microsoft Kinect SDK apps can no longer access the affected sensor interfaces. Closing D.R.A.F.T. VR does not switch the drivers back. In Device Manager, **Kinect** entries under **libusbK USB Devices** are the clue; v2 may instead be named **Xbox NUI Sensor (Composite Parent)**. [OpenKinect documents restoring the SDK driver](https://github.com/OpenKinect/libfreenect2#windows--visual-studio).

<details>
<summary><b>Restore the Microsoft Kinect drivers</b></summary>

1. Close all Kinect apps. Open **Device Manager** and identify your Kinect entries under **libusbK USB Devices**.
2. Unplug the Kinect's USB cable. Choose **View > Show hidden devices** so its entries remain visible.
3. Right-click each affected Kinect entry and choose **Uninstall device**. Select **Delete the driver software for this device** (or **Attempt to remove the driver for this device**, depending on Windows) when offered. Administrator permissions may be needed. Remove only the Kinect entries, not unrelated libusbK devices or USB controllers.
4. If the official SDK is missing, install the matching **SDK 1.8 (v1)** or **SDK 2.0 (v2)** linked above. If already installed but restoration fails, repair/reinstall it.
5. Reconnect the Kinect and choose **Action > Scan for hardware changes**. Restart Windows if requested. Windows should load the Microsoft drivers; the affected Kinect entries should no longer use libusbK. Try **Start** in this app again.

If libusbK returns immediately, its driver package may still be installed. Repeat removal with the driver-removal checkbox selected; do not rerun D.R.A.F.T. VR's libusbK installer during recovery. See [Microsoft's device and driver removal guide](https://learn.microsoft.com/en-us/windows-hardware/drivers/install/using-device-manager-to-uninstall-devices-and-driver-packages).

Switching back to a libusbK-based setup later replaces those drivers again. The same sensor interface cannot use both drivers at once.

</details>

### Kinect v2 repeatedly disconnects on Windows 11

Try disabling audio enhancements: **Sound settings > More sound settings > Recording > Microphone Array (Xbox NUI Sensor) > Properties > Advanced**, then uncheck **Enable audio enhancements** and apply. If Windows presents a separate Enhancements tab, check there instead. Also check the power adapter and use a direct USB 3.0 connection.

<details>
<summary><b>SteamVR trackers are missing or the add-on is blocked</b></summary>

Enable **kinect_fbt** in **SteamVR Settings > Startup / Shutdown > Manage Add-ons**. If absent, close SteamVR and run **Update SteamVR Trackers.cmd** from your installed package, then restart SteamVR.

</details>

<details>
<summary><b>Feet float or clip through the floor</b></summary>

Keep the floor and feet visible, enable **Foot contact**, and check camera alignment and VRChat FBT calibration. **Ankle-to-sole distance** describes your anatomy; it is not a playspace height adjustment.

</details>

For other issues, see the [troubleshooting guide](packaging/windows/TROUBLESHOOTING.md). **Diagnostics** exports numeric tracking information without camera images.

## More information

- [Package guide](packaging/windows/README.md)
- [Build from source](docs/development/BUILD.md) — includes Git LFS and required development dependencies.
- [Third-party notices](THIRD_PARTY_NOTICES.md) and [upstream licenses](docs/licenses/) — includes Fast SAM 3D Body, Meta's models, NVIDIA, PyTorch and OpenVR.
