# KinectAIFusionFBT

**High-Performance, AI-Powered Kinect Full-Body Tracking for SteamVR & VRChat**

[![Native C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![NVIDIA CUDA 12.8](https://img.shields.io/badge/NVIDIA-CUDA%2012.8%20%7C%20TensorRT-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![SteamVR Compatible](https://img.shields.io/badge/SteamVR-Trackers%20%28Waist%20%2B%20Feet%29-black.svg)](https://store.steampowered.com/app/250820/SteamVR/)
[![Platform Windows](https://img.shields.io/badge/Platform-Windows%2010%20%2F%2011%20x64-lightgrey.svg)](https://microsoft.com)

**KinectAIFusionFBT** is a zero-bloat, native Windows C++20 application that transforms your **Kinect for Xbox 360 (v1)** or **Kinect for Xbox One (v2)** into virtual SteamVR Vive Trackers for Full-Body Tracking (FBT) in **VRChat** and other SteamVR titles.

By combining the **Fast SAM 3D Body** deep learning foundation model with high-speed metric depth cloud registration and ground-plane physics, it delivers responsive waist, left foot, and right foot tracking directly on your local GPU—with no Python runtime, no cloud dependency, and no extra wearables required.

---

## Highlights & Features

- **Adaptive SteamVR Smoothing**: Position and rotation filtering runs with SteamVR updates and adjusts to observed movement. It approaches a fixed target without overshoot; live latency and tracking quality depend on camera rate, visibility and PC load.
- **In-Headset SteamVR 3D Calibration Overlay**: A real-time visual guide renders directly in your headset during calibration. Clear 3D controller poses, live countdowns, wrist tracking progress counters (`X/12`), and dedicated retry screens guide you through setup without having to take off your headset.
- **Fast SAM 3D Body Deep Learning**: Selective FP8 TensorRT backbone and native LibTorch C++ GPU decoder reconstruct full anatomical 3D joints from raw camera video in real time.
- **High-Priority GPU Stream**: AI inference uses a dedicated high-priority CUDA stream. It still shares the GPU with VRChat.
- **Adaptive Neural Cadence**: Auto / 30 Hz / 20 Hz / 15 Hz controls the rate of new AI estimates. Depth and foot contacts are processed on every delivered camera frame, including frames that reuse a recent AI estimate.
- **Full Metric Depth & Ground-Plane Locking**: Automatically detects room floor tilt and locks soles to the floor to prevent floating or floor clipping.
- **Playspace Move & Space Drag Support**: Seamlessly moves with OVR Advanced Settings playspace drag and rotation without losing calibration.
- **Saved Alignment Memory**: Calibrate once and click **Confirm saved alignment** in future sessions to jump straight in.

---

## Hardware Compatibility

| Component | Requirements | Notes |
| :--- | :--- | :--- |
| **Kinect Sensor** | **Kinect v1 / Xbox 360** *(Model 1414 / 1473)*<br>**Kinect v2 / Xbox One** *(Model 1520)* | • Kinect v1 requires standard 12V AC power + USB adapter cable (works on USB 2.0 or 3.0).<br>• Kinect v2 requires standard 12V AC power + USB 3.0 adapter cable. |
| **GPU** | **NVIDIA GeForce RTX 4000 / 5000 Series**<br>*(Ada Lovelace or Blackwell)* | **RTX 4000 or 5000 series required** (tested on Windows 11 with RTX 5070 Ti). Requires native hardware FP8 Tensor Cores for the accelerated neural model. Older cards (RTX 20 and 30 series) lack hardware FP8 compute and are not supported. NVIDIA TensorRT and CUDA 12.8 runtimes are bundled in the release. |
| **VR Headset** | **Any SteamVR-compatible PCVR Headset** | Meta Quest (Link / AirLink / Virtual Desktop), Valve Index, HTC Vive, Bigscreen Beyond, Pico 4, Windows Mixed Reality, etc. |
| **Controllers** | **2 Positional VR Controllers** | Used during the quick 5-pose alignment routine. |

---

## Quick Start (Pre-Built Release)

### 1. Prerequisites
1. Install the official Microsoft Kinect driver for your sensor:
   - **For Kinect v1 / Xbox 360**: Install [Microsoft Kinect for Windows SDK 1.8](https://www.microsoft.com/en-us/download/details.aspx?id=40278).
   - **For Kinect v2**: Install [Microsoft Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561).
2. Install the latest [NVIDIA Game Ready Driver](https://www.nvidia.com/Download/index.aspx).
3. Ensure **SteamVR** is installed and has been run at least once.

### 2. Installation
1. Download the latest **`KinectFBT-v1.0-Windows-x64.zip`** from [Releases](https://github.com/LogMeIn-Hamachi/KinectAIFusionFBT/releases).
2. Extract the ZIP into a permanent folder on your PC (e.g. `C:\Tools\KinectAIFusionFBT`).
3. Make sure SteamVR is closed, then right-click **`Install SteamVR Trackers.cmd`** and select **Run as administrator** (or double-click it). This registers the virtual tracker driver with SteamVR.

---

## Updating the SteamVR Driver

Close SteamVR completely, extract the new package to its permanent location, and double-click **Update SteamVR Trackers.cmd** in that package. It switches registration from an older Kinect package to the new bundled driver, leaving old files, settings and unrelated drivers untouched. Start SteamVR afterward. Keep the new folder in place.

If you replace files in the already registered folder, SteamVR loads those files on its next start; the updater confirms that the folder is already registered. It does not download releases or copy preferences between packages. A failed registration update attempts to restore the previous Kinect registration.

## Step-by-Step Calibration Guide

### Step A: Position Your Kinect
- Place the Kinect approximately **1.8 to 2.5 meters (6 to 8 feet)** in front of your play area at roughly waist or chest height.
- Angle the camera so it can see your feet on the floor and your hands raised in front of you.
- *(Kinect v1 only)*: You can use the `−` and `+` motor buttons in the application to electronically adjust camera tilt.

### Step B: Launch & Lock Player
1. Start SteamVR and put on your headset.
2. Open **`KinectRGBD.exe`** and click **Start**.
3. Stand in front of the camera so your body is visible in the preview window, select your body, and click **Lock player**.
   *(Note: The very first launch prepares a local TensorRT GPU engine cache, which takes 30–60 seconds. Subsequent launches start instantly.)*

### Step C: Body Proportions
- Click **Capture proportions**.
- An 8-second countdown will start: stand in a relaxed pose with your feet shoulder-width apart.
- The app collects 4 seconds of bone length samples to fit your exact height and proportions.

### Step D: VR Alignment (In-Headset Guide)
- Reset any OVR Advanced Settings space drag offsets before aligning.
- Put your headset on, grab both controllers, and click **Align to VR** (or press the button in the desktop UI).
- An interactive guide will appear floating in front of you inside SteamVR:
  - **Pose 1 (T-Pose)**: Stand straight, arms extended out to your sides. Squeeze either trigger when steady.
  - **Pose 2 (Forward Reach)**: Hold both controllers straight forward. Squeeze either trigger when steady.
  - **Pose 3 (45° Diagonal)**: Hold controllers diagonally forward. Squeeze either trigger when steady.
  - **Pose 4 (Chest Position)**: Hold controllers near your chest pointing up. Squeeze either trigger when steady.
  - **Pose 5 (Check Pose)**: Hold hands forward **shoulder-width apart, clear of your torso**, pointing forward. Squeeze either trigger to confirm alignment.
  - *If a pose needs adjustment, an amber retry screen will explain what to adjust—simply squeeze your trigger to try that step again without restarting.*

### Step E: Activate Trackers & VRChat
1. In `KinectRGBD.exe`, click **Start trackers**.
2. Look at your SteamVR status window: you will see three new green tracker icons (**Waist**, **Left Foot**, and **Right Foot**).
3. Open **VRChat**:
   - Go to **Settings $\to$ Tracking & IK $\to$ Calibrate FBT**.
   - Stand in T-Pose / I-Pose to match your avatar's trackers, and squeeze both triggers to lock in!

---

## Everyday Use

- **Saved Alignment**: As long as your Kinect has not been moved physically, you do **not** need to re-align every time!
  - Start SteamVR $\to$ Launch `KinectRGBD.exe` $\to$ Click **Start** $\to$ **Lock player** $\to$ **Confirm saved alignment** $\to$ **Start trackers**.
- **Playspace Movement**: Moving your playspace with OVR Advanced Settings space drag works automatically—all three trackers stay physically anchored to your real-world body.

---

## Settings & Tuning

In the **Advanced** tab of `KinectRGBD.exe`:
- **Tracking Cadence**:
  - `Auto (GPU adaptive)` *(Default)*: Starts at up to 30 Hz, reduces the rate after sustained processing delays, and steps back up when the load recovers. It uses processing time and queued-frame delay, not a GPU usage percentage.
  - `30 Hz (Full AI)`: Up to 30 new AI estimates per second.
  - `20 Hz (Balanced)`: Up to 20 new estimates per second, reducing neural work.
  - `15 Hz (Low GPU)`: Up to 15 new estimates per second for demanding VR titles.
  - The displayed actual Hz counts completed AI estimates. All modes are limited by camera delivery and available processing time. A camera delivering 15 fps cannot produce 30 new image estimates per second; selecting 15 Hz does not halve that camera rate again.
  - If tracking seems to drop in and out, click **Diagnostics** after reproducing it. The report separates deliberate AI-result reuse from missing estimates, source switches and tracker validity losses. A temporarily untracked foot stays connected to SteamVR but has an invalid pose until it is observed again.
- **Model Selection**: Switch between **SAM optimized** (fastest, FP8 + TF32) and **SAM original**.
- **OSC Output**: Secondary VRChat output on `localhost:9000`. Uses the same adaptive position/rotation smoothing and bounded prediction as the SteamVR driver, with playspace movement applied after smoothing. Enable OSC in VRChat and select OSC output in the app, then enable output. Expired trackers stop sending updates; VRChat controls how long its last received pose remains visible. App-side smoothing is shared, but receiver timing and tracking-loss behavior can differ from native SteamVR.

---

## Building from Source

### Prerequisites
- Windows 10/11 x64
- [Visual Studio 2022](https://visualstudio.microsoft.com/) with C++ desktop development workload (C++20 support)
- [CMake 3.25+](https://cmake.org/download/)
- [Git with Git LFS](https://git-lfs.com/)
- [NVIDIA CUDA Toolkit 12.8](https://developer.nvidia.com/cuda-toolkit)
- Microsoft Kinect SDK 1.8 and/or SDK 2.0

### Build Steps
```powershell
# 1. Clone the repository with Git LFS
git clone https://github.com/LogMeIn-Hamachi/KinectAIFusionFBT.git
cd KinectAIFusionFBT
git lfs pull

# 2. Configure and build via CMake
cmake --preset windows-release
cmake --build build --config Release

# 3. Run the automated offline test suites (10/10 must pass)
ctest --test-dir build -C Release --output-on-failure
```

---

## Troubleshooting & FAQ

<details>
<summary><b>SteamVR says "Add-on blocked" or trackers don't show up</b></summary>

1. In SteamVR, open **Settings $\to$ Startup / Shutdown $\to$ Manage Add-ons**.
2. Verify that **`kinect_fbt`** is set to **On**.
3. If it is not listed, make sure you ran `Install SteamVR Trackers.cmd` while SteamVR was closed.
</details>

<details>
<summary><b>My feet are clipping into the floor or floating</b></summary>

1. Ensure the Kinect can clearly see the floor where you are standing.
2. In `KinectRGBD.exe`, make sure **Floor Contacts** is checked.
3. Adjust the **Floor height offset** in small 1 cm increments if your avatar uses high heels or shoes.
</details>

<details>
<summary><b>Kinect v1 shows "Kinect SDK 1.8 runtime missing"</b></summary>

You must install the official **Microsoft Kinect for Windows SDK 1.8** installer, not just a bare driver. This provides `Kinect10.dll` in your Windows `System32` directory.
</details>

<details>
<summary><b>Kinect v2 drops frames or fails to start</b></summary>

Kinect v2 requires high USB 3.0 isochronous bandwidth. Ensure the Kinect v2 USB cable is plugged directly into a native motherboard USB 3.0 port (blue port or USB 3.1/3.2), not into an unpowered USB hub.
</details>

<details>
<summary><b>Kinect v2 restarts in a loop / keeps disconnecting on Windows 11</b></summary>

A known Windows 11 driver conflict causes the Kinect v2 to repeatedly power cycle and restart in a continuous loop when audio enhancements are active:
1. Right-click the **speaker icon** in the Windows taskbar (by the clock) and select **Sound settings**.
2. Scroll down and click **More sound settings** to open the classic Sound dialog.
3. Switch to the **Recording** tab.
4. Select **Microphone Array (Xbox NUI Sensor)** and click **Properties**.
5. Switch to the **Advanced** tab.
6. Under **Signal Enhancements**, uncheck **Enable audio enhancements**.
7. Click **Apply**, then **OK**.
</details>

---

## License & Acknowledgements

- Deep learning pose estimation powered by **Fast SAM 3D Body** and **DINOv3** (Meta Platforms, Inc.).
- Native runtime utilizes **NVIDIA TensorRT RTX**, **NVIDIA CUDA Toolkit**, **ONNX Runtime**, and **PyTorch LibTorch C++**.
- Sensor communication via **Microsoft Kinect for Windows SDK**.
- Virtual tracker emulation powered by **Valve OpenVR**.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [docs/licenses/](docs/licenses/) for complete upstream licensing terms.
