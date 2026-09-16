# Troubleshooting

**No sensor:** check the powered adapter and USB connection. V2 requires USB 3 and its Microsoft runtime. Close other applications using the Kinect. Do not replace its USB driver with a generic driver.

**First Start takes a while:** the model is checked and a local GPU cache is prepared. Subsequent starts reuse the cache. Keep the package in a writable folder. The application status reports model preparation/errors. It requires NVIDIA acceleration; there is no CPU SAM fallback.

**Model error:** keep all extracted files together. Install an NVIDIA driver compatible with the bundled CUDA/TensorRT runtimes. Stop tracking and try Advanced → SAM original. If the error persists, use Diagnostics and retain the exact error text. Optimized SAM has been tested on RTX 5070 Ti; other GPUs have not been qualified with this package.

**No trackers in SteamVR:** run the installer from the final extracted location, restart SteamVR, and check that its Kinect add-on is enabled. Lock your player, accept alignment, and press Start trackers. Stop any older copy of the tracker app. When switching between package folders, remove the old registration before installing the new one.

**Trackers move incorrectly during space drag:** first reset OVR space drag/rotation, then confirm saved alignment or Align to VR. After that, dragging and rotating should move all three trackers together. Do not calibrate a saved room against an already-offset playspace. Restarting the app requires confirming the reference again. A SteamVR restart during tracking requires Align to VR again. Changes to the physical camera or room origin need a new alignment. This release targets OVR Advanced Settings' SteamVR standing-space changes; other movers that independently modify device poses are unverified.

**Alignment fails:** keep both wrists visible and separate from your torso, use a consistent normal controller grip and wait until you are comfortable before pressing a trigger/Capture pose. You control when each pose begins. The countdown gives three seconds to settle, followed by sampling. Your head does not need to be visible. Avoid gripping the controllers differently between poses.

**Tracking jitter or wrong side-on/lying poses:** a single camera cannot see occluded limbs. Keep the entire legs in view, avoid furniture covering the feet, and use adequate room lighting for SAM's colour image. Difficult occlusions remain a limitation. The SDK baseline modes in Advanced are diagnostic comparisons; RGB-D fusion is the normal tracking mode.

**Dark v2 colour image:** Advanced → Prioritize 30 fps limits exposure to favour responsiveness. Stop tracking to change it. Automatic exposure can brighten dim scenes but may reduce colour capture to 15 fps. The displayed fps and exposure are measured/reported values, not a guarantee that inference also runs at 30 fps. More room lighting reduces this trade-off.

**GPU load with VRChat:** reduce VRChat's own graphics load to leave processing headroom. SAM optimized is the intended faster model. This preview does not guarantee a specific application frame rate on another GPU.

**Stopped trackers remain visible:** native SteamVR output marks missing or stale poses invalid. The receiver may show the last avatar pose during tracking loss. OSC has different receiver behavior; native SteamVR is the default.

**Diagnostics and recordings:** created only in your own extracted app folder when requested. Review exports before sharing them; they may contain machine paths and sensor information. The public package includes no local session data.
