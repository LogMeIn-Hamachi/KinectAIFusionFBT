#pragma once
#include "sam3d_geometry.hpp"
namespace kf {
struct NlfWarp {
    int pyramidLevel{};
    std::array<V3,3> rotation{};
    std::array<float,9> intrinsics{};
    std::array<V3,3> sourceProjection{};
};
NlfWarp nlfWarp(Crop,const Sam3dCamera &);
std::vector<float> nlfImage(const Frame &,const NlfWarp &);
BodyPrediction nlfDecode(std::span<const float>,const NlfWarp &,const Sam3dCamera &,double host);
}
