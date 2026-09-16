#pragma once
#include "core.hpp"
namespace kf {
// A current image pose and a temporarily predicted metric root are distinct.
// Never creates articulation from an old image or records predicted depth support.
class PoseContinuity {
    V3 root_{}, pending_{};
    double host_{}, depthHost_{}, rms_{}, pendingSince_{}, pendingHost_{};
    uint32_t id_{};
    bool initialized_{}, waiting_{};
public:
    static constexpr double maxDepthAge=.4;
    void reset(){*this={};}
    PosePrior update(const PosePrior&,uint32_t id,double host);
};
}
