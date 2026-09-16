# Third-party notices

Keep this file and docs/licenses with the distributed application. Third-party components retain their own terms; the licenses of source projects do not replace model or NVIDIA binary agreements. Microsoft Kinect installers, SteamVR and NVIDIA display drivers are not bundled.

| Component | Included version / provenance | Included terms in docs/licenses |
|---|---|---|
| Fast SAM 3D Body | yangtiming/Fast-SAM-3D-Body, commit 808b53c7d9c26a7e511d31144f1e5efb058e15c9; native body-only conversion | Fast-SAM-MIT.txt; underlying Meta terms below |
| Meta SAM 3D Body | facebook/sam-3d-body-dinov3, revision 11aaa346c7204874a1cbafe3d39a979080b2c55a; converted encoder and landmark decoder | SAM-checkpoint-LICENSE.txt and SAM-LICENSE.txt |
| DINOv3 | Architecture commit 6876159a11b4df116f30f667f8c9888617df0751; encoder weights from the above checkpoint | DINOv3-LICENSE.txt |
| MHR / Momentum | MHR checkpoint assets; Momentum equations from 11172f05996c1009cae6c5ae8e01654d4f2e270b | MHR-LICENSE.txt (Apache 2.0), Momentum-LICENSE.txt (MIT); converted checkpoint assets also retain checkpoint terms |
| ONNX Runtime | 1.30.0 Windows x64 | ONNXRuntime-LICENSE.txt and ONNXRuntime-ThirdPartyNotices.txt |
| NVIDIA standalone EP | ABI 0.4.1, CUDA 13 build | NVIDIA-EP-LICENSE.txt and NVIDIA-EP-ThirdPartyNotices.txt |
| TensorRT RTX | Runtime/parser/plugins from the official EP release, 1.6 | TensorRT-RTX-SLA.html and TensorRT-RTX-Acknowledgements.txt |
| PyTorch / LibTorch | 2.11.0 CUDA 12.8 native libraries from the official Windows wheel; no Python runtime | PyTorch-LICENSE.txt and PyTorch-NOTICE.txt |
| CUDA, cuDNN and NVIDIA math libraries | Runtime dependencies from the official PyTorch CUDA 12.8 wheel | CUDA-12.8-EULA.html, cuDNN-License.html; notices in PyTorch and NVIDIA texts |
| OpenVR | Headers/client library from 0924064316de3effbcd1acf1e309182a2deb1c05 | OpenVR-LICENSE.txt |
| NuiSensorLib | Microsoft MixedRealityCompanionKit, 0e3e8f3cb44c6683f0a1b0a0f561074fa9e6f88c; linked exposure control | NUISENSOR-LICENSE.txt (MIT) |
| Microsoft Visual C++ | App-local x64 VC143 runtime from Visual Studio's redistribution directory | Microsoft Visual Studio distributable-code terms; Windows Universal CRT is an OS prerequisite |

The optimized model uses selective FP8 conversion of encoder blocks 4–27 and five decoder grid-sampling layout changes. These are derivatives of the same checkpoint, not a newly trained model. Both included model variants remain subject to the SAM and DINOv3 agreements. Source checkpoint provenance and conversion details are in docs/model-provenance; exact distributed hashes are in package-sha256.json and each model's backbone.onnx.files.sha256.

External runtime use is subject to Microsoft's Kinect SDK/runtime terms. Kinect v1/Xbox 360 hardware support under SDK 1.8 has development/test restrictions; this package does not grant broader sensor rights. Kinect v2 uses the installed SDK 2.0/Kinect20 API. SDK headers, import libraries and sensor driver DLLs are not included.

Primary sources:

- [Fast SAM 3D Body](https://github.com/yangtiming/Fast-SAM-3D-Body)
- [Meta checkpoint and agreement](https://huggingface.co/facebook/sam-3d-body-dinov3)
- [TensorRT RTX agreement](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html)
- [Visual Studio redistribution terms/list](https://learn.microsoft.com/en-us/visualstudio/releases/2022/redistribution)
- [Kinect SDK 2.0 agreement](https://download.microsoft.com/download/0/d/c/0dc5308e-36a7-4dcd-b299-b01cdfc8e345/kinect-sdk2.0-eula_en-us.pdf)

NLF, RTMPose, historical model experiments and their weights are not part of this distribution.
