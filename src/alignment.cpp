#include "alignment.hpp"
#include <iomanip>
#include <sstream>
namespace kf {
namespace {
constexpr const char *names[]{"Headset", "Left controller", "Right controller"};
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
V3 medianPoint(const std::deque<AlignmentObservation> &w) {
    std::vector<double> x, y, z;
    for (auto &o : w) {
        x.push_back(o.camera.x);
        y.push_back(o.camera.y);
        z.push_back(o.camera.z);
    }
    return {median(x), median(y), median(z)};
}
} // namespace
void AlignmentSession::reset(bool automaticOffsets,bool controllersOnly) {
    floor_={};
    automaticOffsets_=automaticOffsets;
    controllersOnly_=controllersOnly || automaticOffsets;
    fittedOffsets_={};
    windows_ = {};
    lastAccepted_ = {};
    samples_.clear();
    status_ = {"waiting", "waiting", "waiting"};
    if(controllersOnly_)status_[0]="not used; head visibility is optional";
}
void AlignmentSession::add(const Frame &frame, uint32_t id, const Settings &settings, const PosePrior *prior,const Calibration* reference) {
    const auto vr=reference?calibrationVr(frame.vr,*reference):frame.vr;
    floor_.add(frame.host,frame.floor);
    auto body = std::find_if(frame.bodies.begin(), frame.bodies.end(), [&](auto &b) { return b.id == id; });
    for (int device = 0; device < 3; ++device) {
        if(device==0 && controllersOnly_) {status_[0]="not needed; head may be out of view";continue;}
        auto &window = windows_[device];
        int joint = device == 0 ? Head : device == 1 ? LWrist : RWrist;
        bool observed=body!=frame.bodies.end() && body->joints[joint].confidence>=.7 && finite(body->joints[joint].p);
        V3 point=observed?body->joints[joint].p:V3{};
        // Only depth-supported learned wrists may substitute for a missing SDK
        // wrist; a plausible but hidden network prediction is not calibration evidence.
        if(!observed && device>0 && prior && prior->valid && !prior->depthPredicted && prior->selectedId==id &&
           std::abs(prior->host-frame.host)<.025 && prior->available[joint] && prior->supported[joint] && finite(prior->points[joint])) {
            observed=true;point=prior->points[joint];
        }
        if (!observed) {
            window.clear();
            status_[device] = "Kinect cannot see " + std::string(device == 0 ? "head" : "wrist");
            continue;
        }
        if (!vr.devices[device].valid || std::abs(vr.host - frame.host) > .025 ||
            !finite(vr.devices[device].p) || !std::isfinite(dot(vr.devices[device].q,vr.devices[device].q))) {
            window.clear();
            status_[device] = "no time-matched VR pose";
            continue;
        }
        if (!window.empty() && (frame.host <= window.back().host || frame.host - window.back().host > .12))
            window.clear();
        window.push_back({frame.host, device, point, vr.devices[device],
                          automaticOffsets_?V3{}:settings.deviceOffsets[device]});
        while (window.size() > 1 && frame.host - window.front().host > .4)
            window.pop_front();
        if (window.size() < 6 || frame.host - window.front().host < .25) {
            status_[device] = "hold briefly";
            continue;
        }
        auto center = medianPoint(window);
        V3 worldMean{};
        for (auto &o : window)
            worldMean += o.pair().world;
        worldMean = worldMean / double(window.size());
        std::vector<double> cameraDistances, vrDistances;
        double angle = 0;
        for (auto &o : window) {
            cameraDistances.push_back(norm(o.camera - center));
            vrDistances.push_back(norm(o.pair().world - worldMean));
            angle = std::max(
                angle, 2 * std::acos(std::clamp(std::abs(dot(o.pose.q, window.back().pose.q)), 0.0, 1.0)));
        }
        std::sort(cameraDistances.begin(), cameraDistances.end());
        std::sort(vrDistances.begin(), vrDistances.end());
        size_t tail = (window.size() - 1) * 9 / 10;
        if (cameraDistances[tail] > .035 || vrDistances[tail] > .025 || angle > .25) {
            status_[device] = "moving: pause at this pose";
            continue;
        }
        status_[device] = "collecting steady samples";
        if (frame.host - lastAccepted_[device] < .15 || samples_.size() >= 600)
            continue;
        // Retain an actual paired capture-time sample, not an independently averaged device pose.
        samples_.push_back(window[window.size() / 2]);
        lastAccepted_[device] = frame.host;
    }
}
std::string AlignmentSession::feedback() const {
    std::array<int, 3> counts{};
    for (auto &o : samples_)
        ++counts[o.device];
    std::ostringstream out;
    for (int i = 0; i < 3; ++i)
        out << names[i] << ": " << status_[i] << " (" << counts[i] << ")" << (i == 2 ? "" : "\n");
    return out.str();
}
Calibration AlignmentSession::finish() {
    if(automaticOffsets_) {
        auto floor=floor_.value();
        auto result=calibrateControllers(samples_,fittedOffsets_,floor.valid?&floor:nullptr);
        for(auto &s:samples_)s.offset=fittedOffsets_[s.device];
        return result;
    }
    std::array<int, 3> counts{};
    std::vector<Pair3> pairs;
    for (auto &o : samples_) {
        ++counts[o.device];
        pairs.push_back(o.pair());
    }
    for (int i = 0; i < 3; ++i)
        if (counts[i] < 8) {
            Calibration result;
            result.reason =
                std::string(names[i]) +
                ": too few steady samples. Wear the headset, hold both controllers, and pause in each pose.";
            return result;
        }
    return calibrate(pairs);
}
std::string AlignmentSession::agreement(const Calibration &cal) const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    for (int device = 0; device < 3; ++device) {
        std::vector<double> errors;
        size_t consistent = 0;
        for (auto &sample : samples_)
            if (sample.device == device) {
                auto p = sample.pair();
                double e = norm(cal.transform.apply(p.camera) - p.world);
                errors.push_back(e);
                if (e < calibrationInlierDistance)
                    ++consistent;
            }
        out << names[device] << ": ";
        if(device==0 && controllersOnly_) {out<<"not used; head visibility is optional\n";continue;}
        if (errors.empty())
            out << "no samples";
        else if (cal.spread <= 0)
            out << errors.size() << " samples; fit unavailable";
        else
            out << "median " << median(errors) * 100 << " cm; " << consistent * 100.0 / errors.size()
                << "% within 10 cm";
        if (device < 2)
            out << '\n';
    }
    return out.str();
}
void AlignmentSession::writeCsv(std::ostream &out, const Calibration &cal,bool header) const {
    if(header)out << "host,device,camera_x,camera_y,camera_z,vr_x,vr_y,vr_z,vr_qw,vr_qx,vr_qy,vr_qz,offset_x,offset_y,"
           "offset_z,error_m\n"
        << std::setprecision(12);
    for (auto &s : samples_) {
        out << s.host << ',' << names[s.device];
        for (auto p : {s.camera, s.pose.p})
            out << ',' << p.x << ',' << p.y << ',' << p.z;
        out << ',' << s.pose.q.w << ',' << s.pose.q.x << ',' << s.pose.q.y << ',' << s.pose.q.z << ','
            << s.offset.x << ',' << s.offset.y << ',' << s.offset.z << ',';
        if (cal.spread > 0)
            out << norm(cal.transform.apply(s.camera) - s.pair().world);
        out << '\n';
    }
}
std::string alignmentPoseTitle(int step) {
    constexpr const char* titles[]{"1. Hands at Waist", "2. Hands Apart", "3. Hands at Chest", "4. Reach Forward", "5. Left High, Right Low"};
    return titles[std::clamp(step,0,calibrationPoseCount-1)];
}
std::string alignmentPoseInstruction(int step) {
    constexpr const char* poses[]{
        "Keep elbows by your sides. Hold hands in front of your waist, about shoulder-width apart, with forearms pointing forward.",
        "Keep elbows near your sides. Swing both forearms out to make a shallow V. Hands stay in front of you, a little wider than your shoulders.",
        "Bring hands back to shoulder-width apart. Bend elbows to lift both hands in front of your chest. Keep elbows low and leave space from your chest.",
        "From chest height, reach both hands towards the camera. Stop before your elbows straighten. Keep hands shoulder-width apart.",
        "Check: hold your LEFT hand in front of your chest and your RIGHT hand in front of your waist. Keep hands apart and both clear of your body."};
    return poses[std::clamp(step,0,calibrationPoseCount-1)];
}
AlignmentCue alignmentCue(int step,double elapsed,bool done,bool waiting) {
    AlignmentCue c;c.step=std::clamp(step,0,calibrationPoseCount-1);
    if(done){c.instruction="Alignment finished.";return c;}
    c.instruction="Pose "+std::to_string(c.step+1)+" of 5: "+alignmentPoseInstruction(c.step);
    c.waitingForReady=waiting;
    if(waiting) {
        c.speech=c.instruction+" Take your time. Squeeze either trigger when ready.";return c;
    }
    if(elapsed<0) {
        c.seconds=int(std::ceil(-elapsed));c.speech="Ready. Settle for three seconds.";return c;
    }
    c.retrySeconds=std::max(0,int(std::ceil(20-elapsed)));
    c.collecting=true;c.seconds=std::max(0,int(std::ceil(calibrationHoldSeconds-elapsed)));
    c.speech="Hold still while this pose is captured.";return c;
}
Calibration calibrateKnownOffsets(std::span<const AlignmentObservation> fit,std::span<const AlignmentObservation> validation,const Plane* floor) {
    Calibration result;std::vector<Pair3> pairs;std::array<int,3> counts{};
    auto valid=[](const AlignmentObservation& s){return s.device>=1 && s.device<=2 && s.pose.valid && finite(s.camera) && finite(s.pose.p) && finite(s.offset) && norm(s.offset)<=.30 && std::isfinite(dot(s.pose.q,s.pose.q)) && std::abs(dot(s.pose.q,s.pose.q)-1)<.001;};
    for(auto& s:fit){if(!valid(s)){result.reason="Invalid wrist observation.";return result;}++counts[s.device];pairs.push_back(s.pair());}
    if(counts[1]<12 || counts[2]<12){result.reason="Keep both wrists visible and hold briefly.";return result;}
    result=floor && floor->valid?calibrateWithFloor(pairs,*floor):calibrate(pairs);
    if(!result.valid)return result;
    auto check=[&](std::span<const AlignmentObservation> samples,double limit,int minimum) {
        for(int d=1;d<=2;++d){double sum=0;int n=0,close=0;
            for(auto& s:samples)if(s.device==d){if(!valid(s))return false;double e=norm(result.transform.apply(s.camera)-s.pair().world);sum+=e*e;++n;if(e<.08)++close;}
            if(n<minimum || std::sqrt(sum/n)>limit || double(close)/n<.85)return false;
        }return true;
    };
    if(!check(fit,.04,12)){result.valid=false;result.reason="Wrist positions do not agree closely enough. Keep them clear of your torso and repeat Align to VR.";return result;}
    if(!validation.empty() && !check(validation,.045,8)){result.valid=false;result.reason="The separate check pose did not agree. Keep wrists visible and repeat Align to VR.";return result;}
    result.reason=validation.empty()?"Fit ready; checking a separate pose.":"Alignment checked on a separate pose.";
    if(floor && floor->valid)result.reason+=" Floor height and tilt matched to SteamVR.";
    return result;
}
void GuidedAlignment::reset(double start,const VrSample* vr){*this={};stageStart_=start;if(vr)bindTrackingReference(reference_,*vr);for(auto& s:sessions_)s.reset(true,true);}
size_t GuidedAlignment::size()const{size_t n=0;for(auto& s:sessions_)n+=s.size();return n;}
void GuidedAlignment::capturePose(double time) {
    if(done_ || captureStart_ || !std::isfinite(time) || time<stageStart_)return;
    captureStart_=time+calibrationSettleSeconds;
    sessions_[stage_].reset(true,true);retryReason_.clear();
}
AlignmentProgress alignmentProgress(std::span<const AlignmentObservation> samples,double time) {
    std::array<int,3> count{};std::array<double,3> first{},last{};
    for(const auto& s:samples)if(s.device>=1 && s.device<=2) {
        if(!count[s.device])first[s.device]=s.host;
        ++count[s.device];last[s.device]=s.host;
    }
    AlignmentProgress out;double fraction=1,span=calibrationHoldSeconds;out.ready=true;
    for(int d=1;d<=2;++d) {
        const double covered=count[d]?std::max(0.,last[d]-first[d]):0;
        span=std::min(span,covered);
        fraction=std::min({fraction,count[d]/12.,covered/calibrationHoldSeconds});
        out.ready&=count[d]>=12 && covered>=calibrationHoldSeconds && time-last[d]<=.4 && time>=last[d];
    }
    out.secondsRemaining=int(std::ceil(std::max(0.,calibrationHoldSeconds-span)));
    out.percent=std::clamp(int(100*fraction),0,out.ready?100:99);
    return out;
}
AlignmentCue GuidedAlignment::cue(double time)const {
    auto c=alignmentCue(stage_,captureStart_?time-*captureStart_:0,done_,!captureStart_);
    if(c.collecting) {
        const auto progress=alignmentProgress(sessions_[stage_].samples(),time);
        c.seconds=progress.secondsRemaining;c.progressPercent=progress.percent;
    }
    if(c.waitingForReady && !retryReason_.empty())c.speech=retryReason_+" "+c.speech;
    return c;
}
std::string GuidedAlignment::feedback()const{return done_?result_.reason:!retryReason_.empty()?retryReason_:!captureStart_?"Approximate positions are fine. Keep wrists visible and hold the controllers normally.\nNo capture until you press a trigger or Capture pose.":sessions_[stage_].feedback();}
void GuidedAlignment::writeCsv(std::ostream& out)const{for(int i=0;i<5;++i)sessions_[i].writeCsv(out,candidate_,i==0);}
void GuidedAlignment::add(const Frame& frame,uint32_t id,const Settings& settings,const PosePrior* prior) {
    if(done_)return;
    floor_.add(frame.host,frame.floor);
    bool startRequested=false;
    for(int side=0;side<2;++side) {
        const bool available=std::abs(frame.vr.host-frame.host)<.04 &&
            frame.vr.devices[side+1].valid && frame.vr.triggerAvailable[side];
        if(!available){triggerReleased_[side]=false;continue;}
        const bool pressed=frame.vr.triggerPressed[side];
        startRequested|=pressed && triggerReleased_[side];
        triggerReleased_[side]=!pressed;
    }
    // Each trigger has its own release edge. A held/stuck trigger on one
    // controller must not prevent using the other to retry a failed pose.
    if(startRequested)capturePose(frame.host);
    if(!captureStart_)return;
    const double elapsed=frame.host-*captureStart_;
    auto retry=[&](const std::string& why) {
        captureStart_.reset();triggerReleased_={};retryReason_=why;
        result_.valid=false;result_.reason=why;sessions_[stage_].reset(true,true);
    };
    if(elapsed>20){retry("Could not see both wrists steadily. Adjust your position, then squeeze a trigger to retry this pose. Earlier poses are kept.");return;}
    auto& session=sessions_[stage_];
    if(elapsed<0){session.pause();return;}
    // Use the same observation-selection policy for fit and independent check.
    // SDK wrists are preferred; depth-supported learned wrists remain a fallback.
    if(!reference_.rawReferenceValid && size()==0)bindTrackingReference(reference_,frame.vr);
    session.add(frame,id,settings,prior,&reference_);
    if(elapsed<calibrationHoldSeconds)return;
    // Waiting for visibility is not hold time. Require a real span of paired
    // steady observations and a recent observation of BOTH wrists before moving on.
    if(!alignmentProgress(session.samples(),frame.host).ready)return;
    if(stage_<3){++stage_;stageStart_=frame.host;captureStart_.reset();triggerReleased_={};return;}
    if(stage_==3) {
        std::vector<AlignmentObservation> fit;
        for(int i=0;i<4;++i){const auto& a=sessions_[i].samples();fit.insert(fit.end(),a.begin(),a.end());}
        auto floor=floor_.value();
        candidate_=calibrateControllers(fit,offsets_,floor.valid?&floor:nullptr);
        candidate_.standingToRaw=reference_.standingToRaw;candidate_.rawEpoch=reference_.rawEpoch;
        candidate_.rawReferenceValid=reference_.rawReferenceValid;
        candidate_.referenceSource=reference_.referenceSource;candidate_.referenceUniverse=reference_.referenceUniverse;
        for(int i=0;i<4;++i)sessions_[i].applyOffsets(offsets_);
        if(!candidate_.valid){
            stage_=1;sessions_[2].reset(true,true);sessions_[3].reset(true,true);
            retry("Need clearer controller directions. Repeat hands apart, hands at chest and reach forward; the first pose is kept. "+candidate_.reason);return;
        }
        ++stage_;stageStart_=frame.host;captureStart_.reset();triggerReleased_={};return;
    }
    session.applyOffsets(offsets_);
    // Independent check: never refit the transform on this fifth pose.
    for(int d=1;d<=2;++d) {
        int n=0,close=0;double squared=0;
        for(auto& s:session.samples())if(s.device==d) {
            double error=norm(candidate_.transform.apply(s.camera)-s.pair().world);++n;
            if(error<.12){++close;squared+=error*error;}
        }
        if(!close || double(close)/n<calibrationMinInlierFraction || std::sqrt(squared/close)>calibrationMaxRms) {
            retry(std::string(d==1?"Left":"Right")+" wrist did not agree in the check. Keep your left hand at chest height and right hand at waist height, both in front of you. Squeeze trigger to retry. Earlier poses are kept.");return;
        }
    }
    result_=candidate_;done_=true;
    result_.reason="Alignment accepted: wrist offsets learned and verified on a separate pose. Head visibility was not required.";
}
} // namespace kf
