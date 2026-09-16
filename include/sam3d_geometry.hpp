#pragma once
#include "core.hpp"

namespace kf {
// Fast SAM's GetBBoxCenterScale(1.25), then TopdownAffine(.75), then square.
// Raw image-pixel bbox in; do not feed an already padded RTMPose crop.
Crop sam3dCrop(double x1, double y1, double x2, double y2);
// RGB CHW, [0,1]. Normalization belongs to the exported graph/reference model.
// No image reflection or anatomical label conversion is performed here.
std::vector<float> sam3dImage(const Frame &, Crop);
// Preserve the source's full-image projection and unanchored camera-space 3D.
// A model prediction is not an independently observed metric joint.
struct Sam3dPrediction {
    std::array<V3, 70> cameraPoints{};
    std::array<V2, 70> imagePoints{};
    double host{};
    std::array<M3,2> footRotations{}; // MHR right/left transversetarsal, before camera conversion.
    bool hasFootRotations{};
};
// Positive-focal camera, including the reflection in Kinect's registered RGB.
// Rows map SDK coordinates into image-camera coordinates; do not use a quaternion
// to represent this improper rotation.
struct Sam3dCamera {
    std::array<float,9> intrinsics{};
    std::array<V3,3> rows{};
    V3 translation{};
    bool valid{};
    V3 toSdk(V3 p) const;
};
Sam3dCamera sam3dCamera(const ColorProjection &);
std::array<std::optional<Q>,2> sam3dFootOrientations(const Sam3dPrediction&,const Sam3dCamera&);
// Mapping only: scores intentionally remain zero until visibility is assessed.
Keypoints sam3dLandmarks(const Sam3dPrediction &);
struct Sam3dEvidence {Keypoints keypoints{};PosePrior prior{};};
// Model-independent anatomical Halpe26 points in the RGB camera's coordinates.
struct BodyPrediction {
    std::array<V3,J> cameraPoints{};
    Keypoints landmarks{};
    std::array<bool,J> available{};
    double host{};
};
Sam3dEvidence bodyPoseEvidence(const Frame &,std::uint32_t,const BodyPrediction &,const Sam3dCamera &);
Sam3dEvidence sam3dEvidence(const Frame &,std::uint32_t,const Sam3dPrediction &,const Sam3dCamera &);
} // namespace kf
