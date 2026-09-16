#include "alignment.hpp"
namespace kf {
void StableFloor::add(double host,const Plane& floor) {
    if(!std::isfinite(host) || !floor.valid || !finite(floor.n) || !std::isfinite(floor.d) ||
       std::abs(norm(floor.n)-1)>.01 || floor.n.y<.25 || floor.d<.2 || floor.d>3){samples_.clear();return;}
    if(!samples_.empty() && (host<=samples_.back().first || host-samples_.back().first>.2))samples_.clear();
    samples_.push_back({host,floor});while(samples_.size()>90)samples_.pop_front();
}
Plane StableFloor::value()const {
    if(samples_.size()<12 || samples_.back().first-samples_.front().first<.5)return {};
    V3 n{};double d=0;
    for(auto& [t,p]:samples_){n+=p.n;d+=p.d;}
    n=unit(n);d/=samples_.size();
    for(auto& [t,p]:samples_)if(dot(n,p.n)<std::cos(.025) || std::abs(d-p.d)>.02)return {};
    return {n,d,true};
}
Rigid floorAligned(Rigid r,const Plane& floor) {
    auto n=r.q.rotate(floor.n);V3 up{0,1,0};double c=std::clamp(dot(n,up),-1.,1.);
    auto axis=cross(n,up);Q correction=c<-.999999?axisAngle({1,0,0},pi):normalized(Q{1+c,axis.x,axis.y,axis.z});
    r.q=normalized(correction*r.q);r.t.y=floor.d;return r;
}
Calibration calibrateWithFloor(std::span<const Pair3> pairs,const Plane& floor) {
    auto result=calibrate(pairs);
    if(!floor.valid || !finite(floor.n) || !std::isfinite(floor.d) || std::abs(norm(floor.n)-1)>.01){result.valid=false;result.reason="Invalid floor reference.";return result;}
    if(result.spread<.07)return result;
    Rigid base=floorAligned({},floor),r=base;
    std::vector<double> weights(pairs.size(),1);
    for(int iteration=0;iteration<8;++iteration) {
        double total=0;V3 a{},b{};
        for(size_t i=0;i<pairs.size();++i){a+=base.q.rotate(pairs[i].camera)*weights[i];b+=pairs[i].world*weights[i];total+=weights[i];}
        a=a/total;b=b/total;double cosine=0,sine=0;
        for(size_t i=0;i<pairs.size();++i){auto x=base.q.rotate(pairs[i].camera)-a,y=pairs[i].world-b;
            cosine+=weights[i]*(x.x*y.x+x.z*y.z);sine+=weights[i]*(x.z*y.x-x.x*y.z);}
        if(std::hypot(cosine,sine)<1e-8){result.valid=false;result.reason="Move hands farther apart to determine facing direction.";return result;}
        auto yaw=axisAngle({0,1,0},std::atan2(sine,cosine));r.q=normalized(yaw*base.q);
        r.t=b-yaw.rotate(a);r.t.y=floor.d;
        for(size_t i=0;i<pairs.size();++i)weights[i]=std::min(1.,.035/std::max(norm(r.apply(pairs[i].camera)-pairs[i].world),1e-9));
    }
    std::vector<double> errors;double sum=0;unsigned consistent=0;
    for(auto& p:pairs){double e=norm(r.apply(p.camera)-p.world);errors.push_back(e);if(e<calibrationInlierDistance){sum+=e*e;++consistent;}}
    std::sort(errors.begin(),errors.end());result.transform=r;result.rms=consistent?std::sqrt(sum/consistent):1;
    result.p95=errors[size_t((errors.size()-1)*.95)];
    result.valid=consistent>=pairs.size()*calibrationMinInlierFraction && result.rms<calibrationMaxRms;
    result.reason=result.valid?"Alignment agrees with the floor and wrists.":"Floor and wrists disagree. Check SteamVR floor height and controller grip, then retry alignment.";
    return result;
}
}
