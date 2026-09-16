#include "pose_continuity.hpp"
namespace kf {
PosePrior PoseContinuity::update(const PosePrior& input,uint32_t id,double host) {
    if(!id || !std::isfinite(host)){reset();return {};}
    if(id_!=id || (host_ && (host<=host_ || host-host_>.15)))reset();
    const double dt=host_?host-host_:1./30.;id_=id;host_=host;
    if(input.selectedId!=id || std::abs(input.host-host)>.001 ||
       !input.articulationValid || !input.available[Hip] || !finite(input.points[Hip])) {
        waiting_=false;return {};
    }
    for(int j=0;j<J;++j)if(input.available[j] && !finite(input.points[j])){waiting_=false;return {};}
    auto out=input;
    bool measured=input.valid && !input.depthPredicted && input.rootAnchors>=3 &&
        std::isfinite(input.registrationRms) && input.registrationRms>=0 && input.registrationRms<=.10;
    const V3 candidate=input.points[Hip];
    // Confirm a new root after a prolonged loss, or a discontinuous depth shift.
    if(measured && initialized_ && (host-depthHost_>maxDepthAge || norm(candidate-root_)>.25 || waiting_)) {
        if(!waiting_ || host-pendingHost_>.10 || norm(candidate-pending_)>.12) {
            pending_=candidate;pendingSince_=host;waiting_=true;
        }
        pendingHost_=host;
        if(host-pendingSince_<.10)measured=false;
        else waiting_=false;
    } else if(!measured)waiting_=false;
    if(measured) {
        // Filter one body translation rather than letting every joint choose a
        // different response to the same depth-registration wobble.
        if(!initialized_)root_=candidate;
        else {
            const V3 delta=candidate-root_;
            const double tau=norm(delta)>.06?.02:.05;
            root_+=bounded(delta*(1-std::exp(-dt/tau)),4*dt);
        }
        initialized_=true;depthHost_=host;rms_=input.registrationRms;
        for(auto& p:out.points)p+=root_-candidate;
        out.depthPredicted=false;out.depthAge=0;return out;
    }
    const double age=host-depthHost_;
    if(!initialized_ || age<0 || age>maxDepthAge)return {};
    // Use the fresh image ray for lateral motion, retaining only the last
    // observed depth. Do not reuse the model's uncertain absolute Z/translation.
    const auto ray=candidate-input.cameraOrigin;
    const double denominator=dot(ray,input.cameraForward);
    const double depth=dot(root_-input.cameraOrigin,input.cameraForward);
    if(!std::isfinite(denominator) || denominator<.3 || depth<.4)return {};
    V3 next=input.cameraOrigin+ray*(depth/denominator);
    if(!finite(next) || norm(next-root_)>.35)return {};
    root_+=bounded((next-root_)*(1-std::exp(-dt/.035)),3*dt);
    for(auto& p:out.points)p+=root_-candidate;
    out.valid=true;out.depthPredicted=true;out.depthAge=age;
    out.rootAnchors=0;out.supported={};out.registrationRms=rms_+age*.25;
    return out;
}
}
