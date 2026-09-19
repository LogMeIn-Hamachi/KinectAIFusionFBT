#include "body_tracker.hpp"
namespace kf {
namespace {
double median(std::vector<double> v) {
    std::sort(v.begin(),v.end());return v[v.size()/2];
}
bool usable(const PosePrior& p,int j){return p.available[j] && finite(p.points[j]);}
}
PosePrior BodyTracker::update(const PosePrior& input,uint32_t id,const Frame& f,const Calibration& cal,const Settings& settings) {
    const auto vr=calibrationVr(f.vr,cal);
    const double time=f.host;
    if(!id || !std::isfinite(time)){reset();return {};}
    if(id_!=id)reset();
    else if(host_ && (time<=host_ || time-host_>.3))resetHistory();
    const double dt=host_?time-host_:1./30.;id_=id;host_=time;
    if(input.selectedId!=id || std::abs(input.host-time)>.001 || !input.articulationValid ||
       !usable(input,Hip) || !usable(input,Neck) || !usable(input,LHip) || !usable(input,RHip))return {};
    for(int j=0;j<J;++j)if(input.available[j] && !finite(input.points[j]))return {};
    auto out=input;
    const V3 modelRoot=input.points[Hip];
    auto up=input.points[Neck]-modelRoot,right=input.points[RHip]-input.points[LHip];
    if(norm(up)<.15 || norm(right)<.08 || norm(cross(unit(up),unit(right)))<.3)return {};
    const Q orientation=basis(right,up);
    bool strong=input.valid && !input.depthPredicted && input.rootAnchors>=3 &&
        std::isfinite(input.registrationRms) && input.registrationRms>=0 && input.registrationRms<=.1;
    bool depthObserved=false,vrObserved=false;
    unsigned partialAnchors=0;
    V3 target=modelRoot;
    if(!initialized_) {
        if(!strong)return {};
        root_=modelRoot;started_=time;metricHost_=depthHost_=time;
        depthRms_=input.registrationRms;initialized_=true;
    }
    // A current image ray supplies lateral translation even without metric depth.
    // Never substitute the network's uncalibrated absolute distance.
    V3 ray=input.imageRoot-input.cameraOrigin;
    if(norm(ray)<.1)ray=modelRoot-input.cameraOrigin;
    const double denom=dot(ray,input.cameraForward),z=dot(root_-input.cameraOrigin,input.cameraForward);
    V3 predicted=root_;
    if(denom>.3 && z>.4) {
        // Use image displacement, retaining a learned VR/depth translation.
        // Re-projecting to the absolute ray each frame would erase VR corrections.
        auto direction=ray/denom;
        if(rayReady_)predicted=root_+bounded((direction-previousRay_)*z,3*dt);
        previousRay_=direction;rayReady_=true;
    }
    if(strong) {
        // Require sustained evidence for discontinuities, not for normal motion.
        if(norm(target-predicted)>.25 || pending_) {
            if(!pending_ || norm(target-pendingRoot_)>.12){pendingRoot_=target;pendingSince_=time;pending_=true;}
            if(time-pendingSince_<.1)strong=false;
            else pending_=false;
        }
        if(strong) {
            double quality=std::clamp((.11-input.registrationRms)/.09,.15,1.);
            double tau=.025+.07*(1-quality);
            root_=predicted+bounded((target-predicted)*(1-std::exp(-dt/tau)),3*dt);
            depthObserved=true;depthRms_=input.registrationRms;
        }
    }else pending_=false;
    if(!depthObserved) {
        root_=predicted;
        // Partial depth may gently refine an existing body, never initialize it.
        // Require two agreeing selected-person observations close to prediction.
        V3 sum{};unsigned n{};
        for(unsigned i=0;i<std::min<unsigned>(input.depthRootCount,9);++i) {
            const auto p=input.depthRoots[i];
            if(finite(p) && norm(p-predicted)<.10){sum+=p;++n;}
        }
        if(n>=2) {
            target=sum/double(n);
            root_+=bounded((target-root_)*(1-std::exp(-dt/.16)),.5*dt);
            depthObserved=true;depthRms_=.08;
            partialAnchors=n;
        }
    }
    if(depthObserved){depthHost_=metricHost_=time;}

    // SAM/depth owns body placement. VR may correct only a small horizontal
    // registration bias with two agreeing, depth-supported wrists. Never infer
    // the pelvis from headset position, neck length, or gaze.
    std::array<V3,3> vrTargets{};std::array<bool,3> vrValid{};
    std::array<int,3> vrJoints{Head,LWrist,RWrist};
    if(vrEpoch_ && vr.epoch && vrEpoch_!=vr.epoch){resetHistory();return {};}
    if(vr.epoch)vrEpoch_=vr.epoch;
    if(!cal.valid || (anchorCalibrationReady_ &&
       (norm(cal.transform.t-anchorCalibration_.t)>1e-6 || std::abs(dot(cal.transform.q,anchorCalibration_.q))<.999999))) {
        vrCorrection_={};anchorCalibrationReady_=false;
    }
    if(cal.valid){anchorCalibration_=cal.transform;anchorCalibrationReady_=true;}
    V3 correctionTarget{};
    if(settings.vrConstraints && cal.valid && std::abs(vr.host-time)<.04) {
        for(int d=1;d<3;++d) {
            int j=vrJoints[d];const auto& v=vr.devices[d];
            if(!v.valid || !usable(input,j) || !finite(v.p) || !std::isfinite(dot(v.q,v.q)))continue;
            vrTargets[d]=cal.transform.inverse(v.p+v.q.rotate(settings.deviceOffsets[d]));
            vrValid[d]=finite(vrTargets[d]);
        }
        if(strong && vrValid[1] && vrValid[2] && input.supported[LWrist] && input.supported[RWrist]) {
            V3 left=vrTargets[1]-input.points[LWrist],right=vrTargets[2]-input.points[RWrist];
            V3 delta=(left+right)/2;
            if(norm(left-right)<.06 && norm(delta)<.12) {
                const V3 up=cal.transform.q.conjugate().rotate({0,1,0});
                correctionTarget=bounded(delta-up*dot(delta,up),.035);
            }
        }
    }
    vrCorrection_+=bounded((correctionTarget-vrCorrection_)*(1-std::exp(-dt/1.5)),.015*dt);
    vrObserved=norm(vrCorrection_)>.001;
    // Current RGB articulation can continue for two seconds without a metric
    // correction. Beyond that distance is unobservable; do not hide drift.
    if(time-metricHost_>2.)return {};
    out.valid=true;out.bodyFitted=true;out.vrAnchored=vrObserved;
    out.depthAge=time-depthHost_;out.depthPredicted=!depthObserved;
    out.rootSigma=std::min(.5,.025+depthRms_*.5+(time-metricHost_)*.12);
    out.registrationRms=depthRms_;
    if(!strong){out.rootAnchors=partialAnchors;out.supported={};}
    pelvis_.update(orientation,true,time,pelvis_.initialized?.08:0,1.2,12,.025,true);
    const Q frame=pelvis_.value;
    const V3 filteredRoot=rootFilter_.update(root_+vrCorrection_,time);
    const double pelvisCoherence=norm(pelvis_.motionTrend)/std::max(pelvis_.motionActivity,1e-9);
    const double pelvisSpeed=norm(pelvis_.motionTrend)*std::clamp((pelvisCoherence-.20)/.4,0.,1.);
    // Feet use the camera frame; pelvis filtering must not rotate a planted foot.
    std::array<V3,J> local;
    for(int j=0;j<J;++j)local[j]=orientation.conjugate().rotate(input.points[j]-modelRoot);

    // A disconnected knee/foot branch that suddenly chooses another solution
    // must persist for 100 ms. Coherent ordinary movement is not gated.
    for(int side=0;side<2;++side) {
        int knee=LKnee+side,ankle=LAnkle+side;
        double change=std::max(norm(local[knee]-previousLocal_[knee]),norm(local[ankle]-previousLocal_[ankle]));
        if(worldLegs_[ankle].initialized && (change>.28 || limbPending_[side])) {
            if(!limbPending_[side] || norm(local[ankle]-pendingLocal_[ankle])>.12 || norm(local[knee]-pendingLocal_[knee])>.12) {
                limbPending_[side]=true;limbSince_[side]=time;
                pendingLocal_[ankle]=local[ankle];pendingLocal_[knee]=local[knee];
            }
            if(time-limbSince_[side]<.10) {
                out.footOrientationValid[side]=false;
                for(int j:{knee,ankle,LHeel+side,LToe+side,LSmallToe+side})local[j]=previousLocal_[j];
            } else limbPending_[side]=false;
        }
    }
    // Learn proportions from SAM only. Robust medians settle in the first second;
    // Kinect SDK proportions must not reshape SAM back into its competing pose.
    for(size_t i=0;i<bones.size();++i) {
        const auto [a,b]=bones[i];
        if(!usable(input,a) || !usable(input,b))continue;
        double length=norm(input.points[a]-input.points[b]);
        if(length<.025 || length>.85)continue;
        auto& samples=lengthSamples_[i];
        if(!capturedLengths_ && samples.size()<31 && time-started_<1.2){samples.push_back(length);lengths_[i]=median(samples);}
    }
    for(int j=0;j<J;++j)if(input.available[j]) {
        previousLocal_[j]=local[j];
        bool leg=j==LKnee || j==RKnee || j==LAnkle || j==RAnkle || j>=LToe;
        if(leg) {
            // Filtering the compensating motion in pelvis coordinates caused
            // stationary feet to lag behind a swaying hip. Filter the global
            // foot/knee goal, then let the knee bend around a fixed ankle goal.
            auto target=root_+vrCorrection_+orientation.rotate(local[j]);
            out.points[j]=worldLegs_[j].update(target,time,(j==LKnee || j==RKnee)?.14:.26);
            out.resting[j]=worldLegs_[j].resting;
        } else {
            auto p=local_[j].update(local[j],time);
            out.points[j]=filteredRoot+frame.rotate(p);
            out.resting[j]=rootFilter_.resting && local_[j].resting && pelvisSpeed<.15;
        }
    }
    out.points[Hip]=filteredRoot;
    // Lightweight constrained fit: alternate small device corrections with bone
    // projections. Root stays shared and fixed; wrists affect their arm chain.
    for(int iter=0;iter<6;++iter) {
        for(int d=0;d<3;++d)if(vrValid[d]) {
            int j=vrJoints[d];
            V3 error=vrTargets[d]-out.points[j];
            if(norm(error)<.3)out.points[j]+=error*.35;
        }
        for(size_t i=0;i<bones.size();++i) {
            auto [a,b]=bones[i];double length=lengths_[i];
            if(!input.available[a] || !input.available[b] || length<=0)continue;
            V3 delta=out.points[b]-out.points[a];double n=norm(delta);if(n<1e-6)continue;
            V3 correction=delta*((n-length)/n);
            auto fixed=[](int j){return j==Hip || j==LAnkle || j==RAnkle;};
            const double wa=fixed(a)?0:1,wb=fixed(b)?0:1;
            if(wa+wb==0)continue;
            out.points[a]+=correction*(wa/(wa+wb));out.points[b]+=correction*(-wb/(wa+wb));
        }
    }
    return out;
}
State deliveryState(const State& state,double host) {
    State out=state;
    double age=host-state.host;
    bool any=false;
    for(auto& t:out.trackers) {
        double observed=t.observedHost>0?t.observedHost:state.lastObserved;
        if(age<0 || age>outputHoldSeconds || host-observed>outputHoldSeconds)t.valid=false;
        if(t.valid) {
            t.p+=bounded(t.velocity,4)*std::min(age,.04);
            t.positionSigma+=std::max(0.,age)*.8;any=true;
        }
    }
    if(!any)out.mode=Mode::Degraded;
    else if(age>.06 || out.mode==Mode::Degraded)out.mode=Mode::Predicted;
    return out;
}
}
