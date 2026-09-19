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
    bool capturedLengths_{};
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
    // Transient history belongs to a frame sequence; captured proportions belong
    // to the selected person. Only verified identity recovery may supply a new ID.
    void resetHistory(uint32_t recoveredId=0) {
        const auto lengths=lengths_;const bool captured=capturedLengths_;
        const auto id=recoveredId?recoveredId:id_;
        reset();id_=id;
        if(captured)setLengths(lengths);
    }
    void setLengths(const std::array<double,bones.size()>& lengths) {
        if(!std::all_of(lengths.begin(),lengths.end(),[](double v){return std::isfinite(v) && v>=0 && v<=.85;}))return;
        if(!std::any_of(lengths.begin(),lengths.end(),[](double v){return v>0;}))return;
        lengths_=lengths;capturedLengths_=true;
    }
    bool hasCapturedLengths()const{return capturedLengths_;}
    void restoreLengths(const std::array<double,bones.size()>& lengths,uint32_t id) {
        if(id_!=id){reset();id_=id;}
        setLengths(lengths);
    }
    PosePrior update(const PosePrior&,uint32_t,const Frame&,const Calibration&,const Settings&);
};
// Output delivery tolerates a brief processing stall, not stale moving poses.
// Position extrapolation stops after 40 ms; each tracker expires independently.
inline constexpr double outputHoldSeconds=.24;
State deliveryState(const State&,double host);
}
