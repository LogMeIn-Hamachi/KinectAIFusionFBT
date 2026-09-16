#pragma once
#include "core.hpp"
namespace kf {
// One causal, articulated body. RGB proposes articulation; metric observations
// update translation without selecting an alternative SDK skeleton.
class BodyTracker {
    uint32_t id_{};
    uint64_t vrEpoch_{};
    double host_{}, metricHost_{}, depthHost_{}, depthRms_{}, pendingSince_{};
    V3 root_{}, pendingRoot_{};
    V3 previousRay_{};
    bool rayReady_{};
    bool initialized_{}, pending_{};
    std::array<LearnedPositionFilter,J> local_{};
    std::array<LearnedPositionFilter,J> worldLegs_{};
    std::array<double,bones.size()> lengths_{};
    std::array<std::vector<double>,bones.size()> lengthSamples_{};
    std::array<V3,J> previousLocal_{}, pendingLocal_{};
    std::array<double,2> limbSince_{};
    std::array<bool,2> limbPending_{};
    RotationEvidence pelvis_{};
    LearnedPositionFilter rootFilter_{};
    double started_{};
    V3 vrCorrection_{}; // small horizontal drift correction, never camera-root feedback.
    Rigid anchorCalibration_{};
    bool anchorCalibrationReady_{};
public:
    void reset(){*this={};}
    void setLengths(const std::array<double,bones.size()>& lengths){lengths_=lengths;started_=host_-2;}
    PosePrior update(const PosePrior&,uint32_t,const Frame&,const Calibration&,const Settings&);
};
// Output delivery tolerates a brief processing stall, not stale moving poses.
// Position extrapolation stops after 40 ms; each tracker expires independently.
inline constexpr double outputHoldSeconds=.24;
State deliveryState(const State&,double host);
}
