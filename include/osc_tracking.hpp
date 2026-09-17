#pragma once
#include "steamvr_bridge.hpp"
#include "tracker_smoothing.hpp"
namespace kf {
// Match the native driver's physical-space prediction/filtering, then apply
// the live playspace transform. Never smooth virtual space drag as body motion.
class OscTracking {
    std::array<TrackerSmoothing,3> filters_;
    Rigid reference_{};
    uint32_t body_{};
    uint64_t epoch_{};
    bool initialized_{}, bound_{};
public:
    void reset() { *this=OscTracking{}; }
    std::array<Tracker,3> update(const State& state,const Calibration& calibration,
                              const VrSample& vr,double time) {
        auto cal=calibration;auto space=vr;
        // Retain legacy OSC-only alignment support without inventing a live
        // reference for a calibration that was already bound to SteamVR.
        if(!cal.rawReferenceValid) {
            space={};space.rawTransformValid=true;
            bindTrackingReference(cal,space);
        }
        auto packet=trackerPacket(state,cal,space,time,true);
        if(!validPacket(packet,time)){reset();return {};}
        auto reference=composeRigid(cal.standingToRaw,cal.transform);
        if(!initialized_ || body_!=state.body.id || epoch_!=cal.rawEpoch ||
           bound_!=calibration.rawReferenceValid || norm(reference.t-reference_.t)>1e-8 ||
           angleBetween(reference.q,reference_.q)>1e-7) {
            reset();initialized_=true;reference_=reference;body_=state.body.id;
            epoch_=cal.rawEpoch;bound_=calibration.rawReferenceValid;
        }
        std::array<Tracker,3> result{};
        const auto rawToStanding=inverseRigid(space.standingToRaw);
        for(int i=0;i<3;++i) {
            if(!bridgeTrackingStatus(packet,i,time).valid){filters_[i].reset();continue;}
            const auto& p=packet.poses[i];
            filters_[i].update({p.position[0],p.position[1],p.position[2]},
                {p.rotation[0],p.rotation[1],p.rotation[2],p.rotation[3]},
                p.validUntil-outputHoldSeconds,time);
            auto& out=result[i];out=state.trackers[i];out.valid=true;
            out.p=rawToStanding.apply(filters_[i].position());
            out.q=normalized(rawToStanding.q*filters_[i].rotation());
            out.velocity={};out.angularVelocity={};
        }
        // oscBundle performs the Unity reflection and ZXY Euler conversion once.
        return result;
    }
};
}
