#pragma once
#include "core.hpp"
namespace kf {
// Shared by the actual driver and offline regression tests. Derivatives are
// measured between observations, never from the remaining smoothing error.
class TrackerSmoothing {
    V3 position_{}, lastTarget_{}, velocity_{}, angularVelocity_{};
    Q rotation_{}, lastRotation_{};
    double lastTime_{}, lastObservation_{};
    bool initialized_{};
public:
    void reset() { *this=TrackerSmoothing{}; }
    V3 position() const { return position_; }
    Q rotation() const { return rotation_; }
    double speed() const { return norm(velocity_); }
    void update(V3 targetPos,Q targetRot,double observation,double time) {
        targetRot=normalized(targetRot);
        if(!initialized_ || time-lastTime_>.25 || time<lastTime_ ||
           observation<lastObservation_ || norm(targetPos-position_)>.35) {
            position_=lastTarget_=targetPos;rotation_=lastRotation_=targetRot;
            velocity_=angularVelocity_={};lastObservation_=observation;
            lastTime_=time;initialized_=true;return;
        }
        if(time<=lastTime_)return;
        const double dt=std::min(time-lastTime_,.05);
        if(observation>lastObservation_+1e-6) {
            const double sampleDt=observation-lastObservation_;
            const double alpha=1-std::exp(-sampleDt/.030);
            velocity_=lerp(velocity_,bounded((targetPos-lastTarget_)/sampleDt,4),alpha);
            auto delta=continuous(normalized(targetRot*lastRotation_.conjugate()),Q{});
            V3 axis{delta.x,delta.y,delta.z};double length=norm(axis);
            auto angular=length>1e-9?axis*(2*std::atan2(length,delta.w)/(length*sampleDt)):V3{};
            angularVelocity_=lerp(angularVelocity_,bounded(angular,20),alpha);
            lastTarget_=targetPos;lastRotation_=targetRot;lastObservation_=observation;
        }
        if(time-lastObservation_>.1) {
            velocity_*=std::exp(-dt/.030);angularVelocity_*=std::exp(-dt/.030);
        }
        double fc=std::clamp(6+16*norm(velocity_),6.,28.);
        position_+=(targetPos-position_)*(1-std::exp(-dt*2*pi*fc));
        targetRot=continuous(targetRot,rotation_);
        double fcRot=std::clamp(8+4*norm(angularVelocity_),8.,32.);
        rotation_=normalized(blend(rotation_,targetRot,1-std::exp(-dt*2*pi*fcRot)));
        lastTime_=time;
    }
};
}
